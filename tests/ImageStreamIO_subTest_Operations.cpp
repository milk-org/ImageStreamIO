#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/wait.h>
#include "ImageStreamIO.h"
#include "ImageStreamIO_subTest_Operations.hpp"

// A prefix to all names indicating ImageStreamIO Unit Tests
#define SHM_NAME_PREFIX    "__ISIOUTs__"
#define SHM_NAME_OpsTest SHM_NAME_PREFIX "OpsTest"


////////////////////////////////////////////////////////////////////////
// Forked child data, accessible to SIGCHLD handler
static int child_active = false;
static int child_wstatus = 0;
static pid_t child_pid = 0;


////////////////////////////////////////////////////////////////////////
// Handle signal when forked child exits
static void
sigchld_handler(int sig)
{
    if (!child_active) return;
    int saved_errno = errno;
    errno = 0;
    if (waitpid((pid_t)(-1), &child_wstatus, WNOHANG) == child_pid)
    {
        child_active = false;
    }
    errno = saved_errno;
}


////////////////////////////////////////////////////////////////////////
// Install sigchld_handler(...) action (above) for SIGCHLD signal
static int
install_sigchld_handler()
{
    static int singleton{0};
    if (singleton) { return 0; }

    struct sigaction sa;
    sa.sa_handler = &sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGCHLD, &sa, 0) == -1)
    {
        perror("Error in performing SIGCHLD handler install");
        return -1;
    }
    singleton = 1;
    return 0;
}

int
future_ts(struct timespec delta, struct timespec& future)
{
  error_t save_err = errno;                                // Save errno
  errno = EINVAL;                               // Validate input values
  if (delta.tv_sec < 0) { return -1; }
  if (delta.tv_nsec < 0) { return -1; }
  if (delta.tv_nsec > 999999999) { return -1; }
  if (clock_gettime(CLOCK_TAI, &future) < 0) { return -1; }  // Now time
  errno = save_err;                                     // Restore errno
  future.tv_sec += delta.tv_sec; // Increment current time by delta time
  future.tv_nsec += delta.tv_nsec;
  while (future.tv_nsec > 999999999)      // Handle nanoseconds overflow
  {
    ++future.tv_sec;
    future.tv_nsec -= 1000000000;
  }
  return 0;                                            // Return success
}

#   define WAIT_AS_ns 100000000
#   define WAIT_LIMIT_ns (1000000000 - WAIT_AS_ns)

int
clock_gettime_future(struct timespec& ts)
{
    return future_ts({0,WAIT_AS_ns},ts);
}


////////////////////////////////////////////////////////////////////////
// Operational test of ImageStreamIO library
//
// Each test will increment test_count by 1;
// Each test successfully completed will increment success_count
// If test_count == success_count on return, then all tests succeeded
void
ImageStreamIO_subTest_Operations(int& test_count, int& success_count)
{
    test_count = 0;
    success_count = 0;

    // Install the SIGCHLD handler action
    ++test_count;
    if (install_sigchld_handler()) { return; }
    ++success_count;

    // Open unnnamed pipes to initially communicate with forked child
    // process until ImageStreamIO shmim is ready
    ++test_count;
    int pipes[2];
    if (pipe2(pipes,O_NONBLOCK) < 0) { return; }
    ++success_count;

    // Fork child from paremt
    ++test_count;
    pid_t fork_pid = fork();
    if (fork_pid < 0)
    {
        perror("Error when forking child");
        return;
    }
    ++success_count;

    // Determine which process this is from successful fork
    // Both child and parent processes execute the rest of the code here
    // - The parent will return to the caller, runnin tests until then
    // - The child with exit(...)

    // Child receives 0 returned from fork; parent receives PID of child
    bool i_am_child = fork_pid==0;
    bool i_am_parent = !i_am_child;

    // ImageStreamIO IMAGE structure with shmim confiuration and mapping 
    IMAGE parent_image = { 0 };
    IMAGE child_image = { 0 };

    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    /// Initialize
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

    if (i_am_parent) {

        // Initialize parent:

        // - Count all initialization steps as a single test
        // - Variable [success_count] will not be incremented until and
        //   unless all initialization steps complete successfully
        // - If any step fails befor then, parent will return to caller
        //   with mismatched test and success counts
        // N.B. this pattern will be repeated
        ++test_count;

        // - Module static info for signal handler about forked child
        child_active = true;
        child_pid = fork_pid;

        // - Create shmim; ensure all semaphores' values are zero
        uint32_t dims2[2] = {2, 3};
        uint8_t cpuLocn = -1;
        uint8_t shared = 1;
        if (IMAGESTREAMIO_SUCCESS!=
            ImageStreamIO_createIm_gpu(&parent_image, SHM_NAME_OpsTest
                                      ,2, dims2, _DATATYPE_UINT32
                                      ,cpuLocn, shared, 10, 10
                                      ,MATH_DATA, 0)) { return; }
        //std::cerr
        //<<"Parent inode initial = "
        //<<parent_image.md->inode
        //<<std::endl;

        //   - Reset typical error from _createIm_gpu
        if (errno == ENOENT) { errno = 0; }
        ImageStreamIO_semflush(&parent_image, -1);
        // - Send a synchronization byte to tell forked child to proceed
        if (write(pipes[1],"",1) != 1) { return; }

        // - Close pipes; semaphores can communicate with forked child
        if (close(pipes[1])) { return; }
        if (close(pipes[0])) { return; }

        // - All initialization was successful
        ++success_count;
    }

    if (i_am_child)
    {
        // Initialize forked child:

        // - General pattern is similar to that of parent above,
        //   but child process will exit instead of returning

        // - Wait for single synchronization byte from parent (above)
        char buf[1];
        int iloop=0;
        int iread=0;
        do
        {
            errno = 0;
            iread = read(pipes[0],buf,1);
            usleep(1000);
        } while (iread!=1 && iloop++ < 999);

        // - Close pipes; communicate with parent via shmim semaphores
        close(pipes[0]);
        close(pipes[1]);

        // - Exit if that byte was not received
        if (iread!=1) { exit(1); }

        // - Received byte from parent means shmim is ready; open it
        if (IMAGESTREAMIO_SUCCESS!=
            ImageStreamIO_openIm(&child_image, SHM_NAME_OpsTest))
        {
            perror("Child ImageStreamIO_openIm failed");
            exit(1);
        }
        //std::cerr
        //<<"Child inode initial = "
        //<<child_image.md->inode
        //<<std::endl;

        // - Reset typical error from _openIm
        if (errno == ENOENT) { errno = 0; }
    }

    // Child gets its PID from system call; parent already has child PID
    pid_t my_pid = i_am_child ? getpid() : 0;

    // Lehmer Random Number Generator:  SEED[n+1]=(SEED[n]*A) % M
    // - Use child PID as initial seed
    const uint32_t minstd_m = 1 << 31;
    const uint32_t minstd_a = 48271;
    uint32_t parent_seed = fork_pid % minstd_m;
    uint32_t child_seed = my_pid % minstd_m;

    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    /// Use semaphores for synchronization,
    /// child writes, parent reads, data via shmim over four passes
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////

#   define SEM_TO_CHILD 0
#   define SEM_TO_PARENT 1

    if (i_am_parent)
    {
        // Parent:  read shmim data written by child

        for (int ipass=0; ipass<4; ++ipass)
        {

            if (ipass==3)
            {
                // - Create shmim; ensure all semaphores' values are zero
                uint32_t dims2[2] = {3, 3};
                uint8_t cpuLocn = -1;
                uint8_t shared = 1;
                if (IMAGESTREAMIO_SUCCESS!=
                    ImageStreamIO_createIm_gpu(&parent_image, SHM_NAME_OpsTest
                                              ,2, dims2, _DATATYPE_UINT32
                                              ,cpuLocn, shared, 10, 10
                                              ,MATH_DATA, 0)) { return; }

                //   - Reset typical error from _createIm_gpu
                if (errno == ENOENT) { errno = 0; }
                ImageStreamIO_semflush(&parent_image, -1);
                //std::cerr
                //<<"Parent inode secondary = "
                //<<parent_image.md->inode
                //<<std::endl;
            }

            // - Increment semaphore to trigger child's write pass
            ++test_count;
            if (ImageStreamIO_sempost(&parent_image,SEM_TO_CHILD)
             != IMAGESTREAMIO_SUCCESS)
            {
                perror("Parent sempost to child failed");
                return;
            }
            ++success_count;

            // - Wait for child to update shmim and post other semaphore
            ++test_count;
            struct timespec ts;
            if (future_ts({0,WAIT_AS_ns<<(ipass==3?2:0)},ts))
            {
                perror("Parent clock_gettime failed");
                return;
            }
            errno = 0;
            while (ImageStreamIO_semtimedwait(&parent_image,SEM_TO_PARENT,&ts))
            {
                if (errno==EINTR) { errno = 0; continue; }
                perror("Parent semtimedwait");
                return;
            }
            ++success_count;

            // - Test semlog (posted to by ImageStreamIO_sempost(...))
            if (!ipass)
            {
                // - On first pass by here, both parent and child ...
                int sval = 0;
                ++test_count;
                if (sem_getvalue(parent_image.semlog,&sval))
                {
                    perror("Parent sem_getvalue[semlog]");
                    return;
                }
                //   ... will have each posted once to semlog:
                if (sval != 2)
                {
                    errno = EINVAL;
                    perror("Parent semlog value is not 1");
                    return;
                }
                ++success_count;
            }

            // - Read shmim
            uint32_t* pui32 = parent_image.array.UI32;
            for (uint64_t idat=0; idat<parent_image.md->nelement; ++idat)
            {
                ++test_count;
                if (*(pui32++) == parent_seed) { ++success_count; }
                parent_seed *= minstd_a;
                parent_seed %= minstd_m;
            }

            // - Check seed, at end of previous loop above, against
            //   keyword written by child process
            ++test_count;
            if (strcmp(parent_image.kw[ipass].name,"SEED"))
            {
                errno = EINVAL;
                perror("Parent keyword name is not SEED");
                return;
            }
            if ('L' != parent_image.kw[ipass].type)
            {
                errno = EINVAL;
                perror("Parent keyword type is not 'L'");
                return;
            }
            if (parent_image.kw[ipass].value.numl != parent_seed)
            {
                errno = EINVAL;
                perror("Parent keyword value (seed) does not match");
                return;
            }
            ++success_count;
        }
    }

    if (i_am_child)
    {
        // Child:  write data to shmim, to be read by parent

        for (int ipass=0; ipass<4; ++ipass)
        {
            // - Wait for parent to trigger this pass via semaphore
            struct timespec ts;
            if (clock_gettime_future(ts))
            {
                perror("Child clock_gettime failed");
                exit(101);
            }
            if (ImageStreamIO_semtimedwait(&child_image,SEM_TO_CHILD,&ts))
            {
                // - At this point, the child _semtimedwait failed.
                //
                //   Ideally the cause will be that the semtimedwait
                //   timed out (errno==ETIMEDOUT), becasue the parent
                //   process has opened a new shmim and posted to that
                //   new shmim's child semaphore, and the old shmim's
                //   child semaphore, for which this child was waiting,
                //   was never posted. 
                //
                //   IF THAT IS THE CASE, then
                //   (1) errno will be ETIMEDOUT, and the child's inode
                //       for the old shmim, which is stored as
                //       child_image->md->inode, will no longer be the
                //       inode associated with the new shmim created by
                //       the parent,
                //   AND
                //   (2) the utility ImageStreamIO_check_image_inode()
                //       routine will return the error code
                //         IMAGESTREAMIO_INODE
                //       instead of either IMAGESTREAMIO_SUCCESS or
                //       IMAGESTREAMIO_FAILURE.
                //
                //    If either of those cases is NOT true, then there
                //    is some other reason that the _semtimedwait failed
                //    and the child should exit:
                if (errno!=ETIMEDOUT
                 || IMAGESTREAMIO_INODE!=
                    ImageStreamIO_check_image_inode(&child_image))
                {
                    perror("Child semtimedwait");
                    exit(102);
                }
                //std::cerr
                //<<"Child semtimedwait timed out"
                //<<" due to new shmim during ipass = "
                //<<ipass
                //<<std::endl;

                // - Open the new shmim created by the parent process
                if (IMAGESTREAMIO_SUCCESS!=
                    ImageStreamIO_openIm(&child_image, SHM_NAME_OpsTest))
                {
                    perror("Child ImageStreamIO_openIm on secondary shmim failed");
                    exit(103);
                }
                //std::cerr
                //<<"Child inode secondary = "
                //<<child_image.md->inode
                //<<std::endl;

                // - Reset typical error from _openIm
                if (errno == ENOENT) { errno = 0; }

                // - Wait for the new child semaphore
                if (clock_gettime_future(ts))
                {
                    perror("Child clock_gettime failed");
                    exit(104);
                }
                if (ImageStreamIO_semtimedwait(&child_image,SEM_TO_CHILD,&ts))
                {
                    perror("Child secondary semtimedwait");
                    exit(105);
                }
            } // if (ImageStreamIO_semtimedwait(&child_image,SEM_TO_CHILD,&ts))

            // - Update shmim
            uint32_t* pui32 = child_image.array.UI32;
            for (uint64_t idat=0; idat<child_image.md->nelement; ++idat)
            {
                *(pui32++) = child_seed;
                child_seed *= minstd_a;
                child_seed %= minstd_m;
            }

            // - Write seed, at end of previous loop above, to keyword
            strncpy(child_image.kw[ipass].name,"SEED",5);
            child_image.kw[ipass].type = 'L';
            child_image.kw[ipass].value.numl = child_seed;

            // - Increment semaphore to trigger parrents's read pass
            if (ImageStreamIO_sempost(&child_image,SEM_TO_PARENT)
             != IMAGESTREAMIO_SUCCESS)
            {
                perror("Child sempost to parent");
                exit(105);
            }
        }
    }

    if (i_am_child) { exit(0); }

    if (i_am_parent)
    {
        ++test_count;
        for (int ims=0; child_active && ims<999; ++ims)
        {
           usleep(1000);
           if (errno==EINTR) { errno = 0; }
        }
        if (child_active)
        {
            std::cerr << "Child{PID=" << child_pid << "] still active"
            << std::endl;
            return;
        }
        if (!WIFEXITED(child_wstatus))
        {
            std::cerr << "Child{PID=" << child_pid << "] did not exit()"
            << std::endl;
            return;
        }
        if (WEXITSTATUS(child_wstatus))
        {
            std::cerr << "Child{PID=" << child_pid
            << "] exited with status ["
            << WEXITSTATUS(child_wstatus)
            << "]"
            << std::endl;
            return;
        }
        ++success_count;
    }

    return;
}

/*
 * Build standalone from source in this repo:
 *
 *   g++ -D__ISIOUT_OPS_TEST_MAIN__=main      \
 *       -I..                                 \
 *       ImageStreamIO_subTest_Operations.cpp \
 *       ../ImageStreamIO.c                   \
 *       -pthread                             \
 k       -o ImageStreamIO_subTest_Operations
 *
 * Build standalone against ImageStreamIO shared library and header file
 * under /usr/local/:
 *
 *   g++ -D__ISIOUT_OPS_TEST_MAIN__=main      \
 *       -I/usr/local/include/ImageStreamIO   \
 *       ImageStreamIO_subTest_Operations.cpp \
 *       -lImageStreamIO -pthread             \
 *       -o ImageStreamIO_subTest_Operations
 */
int
__ISIOUT_OPS_TEST_MAIN__(int argc, char** argv)
{
    int success_count;
    int test_count;
    ImageStreamIO_subTest_Operations(test_count, success_count);
    std::cerr
    << "Success on " << success_count << " out of " << test_count
    << " tests; "
    << (success_count == test_count ? "Success" : "Failure")
    << std::endl;

    return (test_count - success_count);
}
