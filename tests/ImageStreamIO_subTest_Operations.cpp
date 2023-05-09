#include <cerrno>
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
#   define WAIT_AS_ns 100000000
#   define WAIT_LIMIT_ns (1000000000 - WAIT_AS_ns)

    if (i_am_parent)
    {
        // Parent:  read shmim data written by child

        for (int ipass=0; ipass<4; ++ipass)
        {
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
            if (clock_gettime(CLOCK_REALTIME,&ts))
            {
                perror("Parent clock_gettime failed");
                return;
            }
            if (ts.tv_nsec < WAIT_LIMIT_ns) {
                ts.tv_nsec += WAIT_AS_ns;
            }
            else
            {
                ts.tv_nsec -= WAIT_LIMIT_ns;
                ++ts.tv_sec;
            }
            if (ImageStreamIO_semtimedwait(&parent_image,SEM_TO_PARENT,&ts))
            {
                perror("Parent semtimedwait");
                return;
            }
            ++success_count;

            // - Read shmim
            uint32_t* pui32 = parent_image.array.UI32;
            for (uint64_t idat=0; idat<parent_image.md->nelement; ++idat)
            {
                ++test_count;
                if (*(pui32++) == parent_seed) { ++success_count; }
                parent_seed *= minstd_a;
                parent_seed %= minstd_m;
            }
        }
    }

    if (i_am_child)
    {
        // Child:  write data to shmim, to be read by parent

        for (int ipass=0; ipass<4; ++ipass)
        {
            // - Wait for parent to trigger this pass via semaphore
            struct timespec ts;
            if (clock_gettime(CLOCK_REALTIME,&ts) == -1)
            {
                perror("Child gettime");
                exit(100);
            }
            if (ts.tv_nsec < WAIT_LIMIT_ns) {
                ts.tv_nsec += WAIT_AS_ns;
            }
            else
            {
                ts.tv_nsec -= WAIT_LIMIT_ns;
                ++ts.tv_sec;
            }
            if (ImageStreamIO_semtimedwait(&child_image,SEM_TO_CHILD,&ts))
            {
                perror("Child semtimedwait");
                exit(101);
            }

            // - Update shmim
            uint32_t* pui32 = child_image.array.UI32;
            for (uint64_t idat=0; idat<child_image.md->nelement; ++idat)
            {
                *(pui32++) = child_seed;
                child_seed *= minstd_a;
                child_seed %= minstd_m;
            }

            // - Increment semaphore to trigger parrents's read pass
            if (ImageStreamIO_sempost(&child_image,SEM_TO_PARENT)
             != IMAGESTREAMIO_SUCCESS)
            {
                perror("Child sempost to parent");
                exit(102);
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
