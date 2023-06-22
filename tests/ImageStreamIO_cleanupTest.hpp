#ifndef __IMAGESTREAMIO_CLEANUPTEST_HPP__
#define __IMAGESTREAMIO_CLEANUPTEST_HPP__
#include <cerrno>
#include <vector>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <csignal>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>
#include "../ImageStreamIO.h"

/* *********************************************************************
class:  ISIO_CLEANUP

- Test of ImageStreamIO cleanup framework

PARENT                             CHILD
======                             =====

Construct ISIO_CLEANUP

Remove shmim file, if present

Block USR2 signal

Fork child process  >===========>  Start of child process
                                     - N.B. exit on any error
                                   Remove (unlikely) SEM files
                                   Check for shmim file
                                   Create shmim
                                   Flush semaphores

                                      |
                                      v

Wait for USR2 signal  <=========<  Send USR2 signal to parent process

Open shmim

Check for semaphore (sem) files
- error if:
  - found with unnamed sems
  - not found with named sems

    |
    v

Post semptr[0]        >=========>  Do timed wait for semptr[0]

                                      |
                                      v

watch semptr[1] to    <=========<  Post semptr[1]
  detect when its
  value becomes 1

    |
    v

EITHER kill CHILD     >=========>  EITHER child process dies when killed
  (kill -KILL)
  (kill -9   )

OR do timed wait      >=========>  OR watch semptr[1] to detect
  for semptr[1]                       when its value becomes 0

                                      |
                                      v

                                   Destroy shmim
                                     |
                                     +-> destroySem
                                         |
                                         +-> sem_unlink removes sem file

                                      |
                                      v

Wait for CHLD signal  <=========<  Normal exit via exit(0)
- also detects child
  dying when killed
  above
    |
    v
Check for shmim & sem files
- error if found when child
  was not killed above

Remove any shmim or sem files

 * ********************************************************************/
class ISIO_CLEANUP
{
private:
    static const std::string shmim_prefix;
    static std::string shmim_name;
public:
    bool using_unnamed_sem() { return sizeof(SEMFILEDATA) <= STRINGMAXLEN_SEMFILENAME; }
    const std::string& get_shmim_name() const { return ISIO_CLEANUP::shmim_name; }

private:
    std::string shmim_filepath{""};
    std::string sem_prefix{""};
    bool failed{true};
    pid_t parent_pid{0};
    pid_t forked_child_pid{0};
    IMAGE image{0};
    enum
    { ISEM_PARENT_TO_CHILD = 0
    , ISEM_CHILD_TO_PARENT
    , ISEM_LAST
    };
public:
    ISIO_CLEANUP();
    ~ISIO_CLEANUP() noexcept;
    void _constructor();
    void _destructor();

    const std::string& get_shmim_filepath() const { return shmim_filepath; }
    const std::string& get_sem_prefix() const { return sem_prefix; }
    bool get_failed() { return failed; }

    // Parent routines
    std::string rm_shmim_filepath_01();
    std::string block_SIGUSR2_02(bool);
    std::string fork_child_03(int which = 0);
    std::string wait_for_SIGUSR2_04();
    std::string open_shmim_05();
    std::string check_for_semfiles_06();
    std::string release_the_child_07();
    std::string wait_for_sem_08(bool);
    std::string close_shmim_09();
    std::string wait_for_child_10(bool);
    std::string file_cleanup_11(bool);

    // Child routines
    void run_child_sequence(int which);
    void run_child_sequence_shmim();
    void run_child_sequence_local();

    // Utility routines
    int find_semfiles(std::vector<std::string>&);
};

// Initialize static members
const std::string ISIO_CLEANUP::shmim_prefix = std::string("isio_cleanup_test_");
std::string ISIO_CLEANUP::shmim_name = std::string(ISIO_CLEANUP::shmim_prefix);

// Destructor is wrapper for _destructor, which resets instance
ISIO_CLEANUP::~ISIO_CLEANUP() { _destructor(); }
void
ISIO_CLEANUP::_destructor()
{
    // Restore the shmim name
    ISIO_CLEANUP::shmim_name.assign(ISIO_CLEANUP::shmim_prefix);
    failed = false;
    block_SIGUSR2_02(false);
    parent_pid = forked_child_pid = 0;
    memset(&image,0,sizeof image);
    return;
}

// Constructor is wrapper for _contructor, which sets up instance
ISIO_CLEANUP::ISIO_CLEANUP() { _constructor(); }
void
ISIO_CLEANUP::_constructor()
{
    // Assume failure
    failed = true;

    // If another instance of this class has already modified
    // the shmim name, then return with failed status
    if (shmim_name.length() != shmim_prefix.length()) { return; }

    char t17[2*(1+sizeof(time_t))]{0};
    char fmt10[20]{0};
    sprintf(fmt10,"%%0%dlx",(int)(sizeof t17)-1);
    sprintf(t17,fmt10,time((time_t*)0));
    shmim_name += t17;

    char c_shmim_filepath[STRINGMAXLEN_FILE_NAME]{0};
    char c_sem_prefix[STRINGMAXLEN_FILE_NAME+4]{"sem."};

    if (int rtn=ImageStreamIO_filename(c_shmim_filepath,STRINGMAXLEN_FILE_NAME,shmim_name.c_str()))
    {
        // If the call above fails to create a valid filepath,
        // then return with failed status
        std::cerr
        << "[" << rtn << "]=failed ImageStreamIO_filename(...) result" << std::endl
        ;
        return;
    }

    // Append shmim filepath to semaphore prefix, replacing slashes with
    // full-stops, and stopping at first full-stop i.e. at .im.shm
    char c;
    char* pc_shmim_filepath{c_shmim_filepath};
    char* pc_sem_prefix{c_sem_prefix+strlen(c_sem_prefix)};
    do
    {
        c = *(pc_shmim_filepath++);
        *pc_sem_prefix = c=='/' ? '.' : (c=='.' ? '\0' : c);
    } while (*(pc_sem_prefix++));

    // Copy local char* C-strings to std::string copies
    shmim_filepath.assign(c_shmim_filepath);
    sem_prefix.assign(c_sem_prefix);

    // Indicate this instance of the class is okay
    failed = false;
}

// /////////////////////////////////////////////////////////////////////
// ISIO_CLEANUP parent routines
// - All return std::stirng("OK") on success
// /////////////////////////////////////////////////////////////////////

// Remove/unlink SHMIM filepath entry
std::string
ISIO_CLEANUP::rm_shmim_filepath_01()
{
    if (failed) { return std::string("rm_shmim_filepath_01:  failed a previous step"); }

    // Create in case it does not already exist
    int fd = open(shmim_filepath.c_str(),O_RDWR|O_CREAT, 0660);
    if (-1 == fd)
    {
        failed = true;
        return "Failed to open SHMIM file[" + shmim_filepath
             + "]{"
             + strerror(errno)
             + "}"
             ;
    }
    // Close it
    if (close(fd))
    {
        failed = true;
        return "Failed to close SHMIM file[" + shmim_filepath
             + "]{"
             + strerror(errno)
             + "}"
             ;
    }
    // Remove it
    if (-1 == unlink(shmim_filepath.c_str()))
    {
        failed = true;
        return "Failed to unlink SHMIM file[" + shmim_filepath
             + "]{"
             + strerror(errno)
             + "}"
             ;
        ;
    }
    return "OK";
}

std::string
ISIO_CLEANUP::block_SIGUSR2_02(bool block)
{
    if (failed) { return std::string("block_SIGUSR2_02:  failed a previous step"); }

    // Construct signal set with only SIGUSR2 in it
    sigset_t sigset;
    if (-1 == sigemptyset(&sigset))
    {
        failed = true;
        return std::string("block_SIGSUR2_02:  failed to empty signal set{")
             + strerror(errno)
             + "}"
             ;
    }
    if (-1 == sigaddset(&sigset, SIGUSR2))
    {
        failed = true;
        return std::string("block_SIGSUR2_02:  failed to add SIGUSR2 to signal set{")
             + strerror(errno)
             + "}"
             ;
    }

    // Block or unblock SIGUSR2
    if(-1 == sigprocmask(block ? SIG_BLOCK : SIG_UNBLOCK, &sigset, nullptr))
    {
        failed = true;
        return std::string("block_SIGSUR2_02:  failed to block/unblock SIGUSR2{")
             + strerror(errno)
             + "}"
             ;
    }
    return "OK";
}

std::string
ISIO_CLEANUP::fork_child_03(int which /* = 0 */)
{
    if (failed) { return std::string("fork_child_03:  failed a previous step"); }

    if (forked_child_pid || parent_pid)
    {
        failed = true;
        return std::string("A child process was already forked");
    }

    parent_pid = getpid();

    if (-1 == (forked_child_pid=fork()))
    {
        failed = true;
        return std::string("Failed to fork child process{")
             + strerror(errno)
             + "}"
             ;
    }
    if (!forked_child_pid)
    {
        // This is now the child process
        forked_child_pid = getpid();
        run_child_sequence(which);
        // We should never get to here
        exit(100);
    }
    // This is still the parent process
    return "OK";
}

std::string
ISIO_CLEANUP::wait_for_SIGUSR2_04()
{
    if (failed) { return std::string("wait_for_SIGUSR2_04:  failed a previous step"); }

    // Construct signal set with only SIGUSR2 in it
    sigset_t sigset;
    if (-1 == sigemptyset(&sigset))
    {
        failed = true;
        return std::string("wait_for_SIGUSR2_04:  Failed to empty signal set{")
             + strerror(errno)
             + "}"
             ;
    }
    if (-1 == sigaddset(&sigset, SIGUSR2))
    {
        failed = true;
        return std::string("wait_for_SIGUSR2_04:  failed to add SIGUSR2 to signal set{")
             + strerror(errno)
             + "}"
             ;
    }

    // Wait up to 10s to receive a SIGUSR2
    siginfo_t si;
    timespec ts{10,0};
    int isigcheck = sigtimedwait(&sigset,&si,&ts);
    if(-1 == isigcheck)
    {
        failed = true;
        return std::string("wait_for_SIGUSR2_04:  failed sigtimedwait call{")
             + strerror(errno)
             + "}"
             ;
    }
    return "OK";
}

std::string
ISIO_CLEANUP::open_shmim_05()
{
    if (failed) { return std::string("open_shmim_05:  failed a previous step"); }

    // Received SIGUSR2 from child, so image is ready; connect to it
    if (IMAGESTREAMIO_SUCCESS != ImageStreamIO_openIm(&image, shmim_name.c_str()))
    {
        failed = true;
        return std::string("open_shmim_05:  failed ImageStreamIO_openIm{")
             + shmim_name
             + "}"
             ;
    }
    return "OK";
}

std::string
ISIO_CLEANUP::check_for_semfiles_06()
{
    if (failed) { return std::string("check_for_semfiles_06:  failed a previous step"); }

    std::vector<std::string> vsemfiles{};
    if (find_semfiles(vsemfiles))
    {
        failed = true;
        return std::string("check_for_semfiles_06:  failed to get semaphore filenames{")
             + strerror(errno)
             + "}"
             ;
    }
    if (using_unnamed_sem())
    {
        if (!vsemfiles.empty())
        {
            failed = true;
            return std::string("check_for_semfiles_06:  at least one named-semaphore file is present{")
                 + vsemfiles[0]
                 + "}"
                 ;
        }
    }
    else
    {
        if (vsemfiles.empty())
        {
            failed = true;
            return std::string("check_for_semfiles_06:  no named-semaphore files are present{/dev/shm/")
                 + sem_prefix
                 + "*}"
                 ;
        }
    }

    return "OK";
}

std::string
ISIO_CLEANUP::release_the_child_07()
{
    if (failed) { return std::string("release_the_child_07:  failed a previous step"); }

    // Post semaphore, on which child is waiting
    if (IMAGESTREAMIO_SUCCESS != ImageStreamIO_sempost(&image, ISEM_PARENT_TO_CHILD))
    {
        failed = true;
        return std::string("release_the_child_07:  failed to post a semaphore{")
             + strerror(errno)
             + "}"
             ;
    }

    return "OK";
}

std::string
ISIO_CLEANUP::wait_for_sem_08(bool kill_child)
{
    if (failed) { return std::string("wait_for_sem_08:  failed a previous step"); }

    int sv{0};
    int icount{0};
    errno = 0;
    do
    {
        // Wait up to 5s while semaphore has a value of 0
        if (!(sv=ImageStreamIO_semvalue(&image,ISEM_CHILD_TO_PARENT)))
        {
            usleep(50000);  // Sleep for 100ms
        }
    } while (++icount<100 && !sv && !errno);   // Wait at least 5s

    if (sv != 1)
    {
        failed = true;
        return std::string("wait_for_sem_08:  semaphore did not increment{")
             + (errno ? strerror(errno) : "unknown reason")
             + "}"
             ;
    }

    if (kill_child)
    {
        // EITHER kill the child which posted the semaphore and is
        // waiting for its value to drop to 0 ...
        errno = 0;
        if (kill(forked_child_pid, SIGKILL))
        {
            failed = true;
            return std::string("wait_for_sem_08:  failed to kill child{")
                 + strerror(errno)
                 + "}"
                 ;
        }
    }
    else
    {
        // ... OR decrement the semaphore
        if (ImageStreamIO_semtrywait(&image,ISEM_CHILD_TO_PARENT))
        {
            failed = true;
            return std::string("wait_for_sem_08:  failed tryway on semaphore{")
                 + (errno ? strerror(errno) : "unknown reason")
                 + "}"
                 ;
        }
    }

    return "OK";
}

std::string
ISIO_CLEANUP::close_shmim_09()
{
    if (failed) { return std::string("close_image_09:  failed a previous step"); }

    // Close the shmim
    if (IMAGESTREAMIO_SUCCESS != ImageStreamIO_closeIm(&image))
    {
        failed = true;
        return std::string("close_image_09:  failed to close image{")
             + (errno ? strerror(errno) : "unknown reason")
             + "}"
             ;
    }

    return "OK";
}

std::string
ISIO_CLEANUP::wait_for_child_10(bool kill_child)
{
    //if (failed) { return std::string("wait_for_child_10:  failed a previous step"); }

    int wp{0};
    int wstatus{0};
    int icount{0};
    errno = 0;
    do
    {
        // Wait up to 5s for child to exit or die
        if (forked_child_pid != (wp=waitpid(forked_child_pid,&wstatus,WNOHANG)))
        {
            if (icount>99)
            {
                int save_errno = errno;
                kill(forked_child_pid, SIGTERM);
                errno = save_errno;
            }
            usleep(50000);  // Sleep for 100ms
        }
    } while (++icount<105 && wp!=forked_child_pid);   // Wait at least 5s

    if (wp != forked_child_pid)
    {
        failed = true;
        return std::string("wait_for_child_10:  failed to detect child change of state{")
             + ((wp==-1 && errno) ? strerror(errno) : "unknown reason")
             + "}"
             ;
    }

    if (WIFEXITED(wstatus))
    {
        int child_exit_status = WEXITSTATUS(wstatus);
        if (child_exit_status || kill_child)
        {
            failed = true;
            return ( kill_child
                   ? std::string("wait_for_child_10:  SIGKILLed child should not have exited, but did{")
                   : std::string("wait_for_child_10:  child exited with non-zero status{")
                   )
                 + std::to_string(child_exit_status)
                 + "}"
                 ;
        }
    }
    else if (WIFSIGNALED(wstatus))
    {
        int child_signaled_value = WTERMSIG(wstatus);
        if (child_signaled_value != SIGKILL || !kill_child)
        {
            char* psignal = strsignal(child_signaled_value);
            failed = true;
            return std::string("wait_for_child_10:  child terminated with signal{")
                 + (psignal ? psignal : "Unknown signal")
                 + "}"
                 ;
        }
    }

    return "OK";
}

std::string
ISIO_CLEANUP::file_cleanup_11(bool kill_child)
{
    if (failed) { return std::string("file_cleanup_11:  failed a previous step"); }

    std::vector<std::string> vsemfiles{};
    if (find_semfiles(vsemfiles))
    {
        failed = true;
        return std::string("file_cleanup_11:  failed to get semaphore filenames{")
             + strerror(errno)
             + "}"
             ;
    }

    std::string leftover{""};

    int leftover_count{0};

    int fd = open(shmim_filepath.c_str(),O_RDONLY);
    close(fd);
    errno = 0;
    if (fd > -1)
    {
        leftover.assign(shmim_filepath);
        ++leftover_count;
        unlink(shmim_filepath.c_str());
    }

    while (!vsemfiles.empty())
    {
        leftover.assign(vsemfiles.back());
        ++leftover_count;
        vsemfiles.pop_back();
        unlink(leftover.c_str());
    }
    errno = 0;

    if (leftover_count && !kill_child)
    {
        failed = true;
        return std::string("file_cleanup_11:  at least one file was not cleaned up{")
             + leftover
             + "}"
             ;
    }
    else if (leftover_count)
    {
        std::cerr
        << leftover_count
        << ","
        << leftover
        << "=leftover_count,last_leftover_file"
        << std::endl;
    }

    return "OK";
}

// /////////////////////////////////////////////////////////////////////
// ISIO_CLEANUP child sequences
// /////////////////////////////////////////////////////////////////////
void
ISIO_CLEANUP::run_child_sequence(int which)
{
    switch (which)
    {
        case 0:
            run_child_sequence_shmim();
            break;
        case 1:
            run_child_sequence_local();
            break;
        default:
            std::cerr
            << "ISIO_CLEANUP::run_child_sequence:  invalid child sequence{"
            << strerror(which)
            << "}"
            << std::endl;
            break;
    }
}

// - ISIO_CLEANUP child sequence that works with a shmim file
void
ISIO_CLEANUP::run_child_sequence_shmim()
{
    // Clean up any semaphore files
    std::vector<std::string> vsemfiles{};
    if (find_semfiles(vsemfiles))
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to get semaphore filenames{"
        << strerror(errno)
        << "}"
        << std::endl;
        exit(1);
    }
    while (!vsemfiles.empty())
    {
        unlink(vsemfiles.back().c_str());
        vsemfiles.pop_back();
    }

    // Check that shmim file does not exist
    int fd = open(shmim_filepath.c_str(),O_RDONLY);
    if (fd > -1 || errno != ENOENT)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  found shmim file that should not exist{"
        << shmim_filepath
        << "}"
        << std::endl;
        exit(2);
    }
    close(fd);
    errno = 0;

    // Create shmim
    uint32_t dims[2] = {2,3};
    errno_t createIm =
    ImageStreamIO_createIm_gpu(&image, shmim_name.c_str()
                              ,(sizeof dims) / (sizeof *dims), dims, _DATATYPE_FLOAT
                              ,-1, 1, ISEM_LAST, 10, MATH_DATA,0);
    if (IMAGESTREAMIO_SUCCESS != createIm)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed createIm{"
        << std::to_string(createIm)
        << "}"
        << std::endl;
        exit(3);
    }

    // Ensure semaphore values are zero
    ImageStreamIO_semflush(&image,ISEM_PARENT_TO_CHILD);
    ImageStreamIO_semflush(&image,ISEM_CHILD_TO_PARENT);

    // Send USR2 signal to parent
    if (-1 == kill(parent_pid, SIGUSR2))
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to send SIGUSR2 to parent{"
        << strerror(errno)
        << "}"
        << std::endl;
        exit(4);
    }

    // Wait for semaphore from parent indicating parent has opened shmim
    struct timespec semwts{0,0};
    if (clock_gettime(CLOCK_REALTIME, &semwts) == -1)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to get a time from clock_gettime{"
        << (errno ? strerror(errno) : "unknown error")
        << "}"
        << std::endl;
        exit(5);
    }
    semwts.tv_sec += 5;
    if (ImageStreamIO_semtimedwait(&image,ISEM_PARENT_TO_CHILD,&semwts))
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to get semaphore from parent{"
        << (errno ? strerror(errno) : "unknown error")
        << "}"
        << std::endl;
        exit(5);
    }

    // Write response semaphore for parent
    if (IMAGESTREAMIO_SUCCESS != ImageStreamIO_sempost(&image,ISEM_CHILD_TO_PARENT))
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to write semaphore for parent{"
        << (errno ? strerror(errno) : "unknown error")
        << "}"
        << std::endl;
        exit(6);
    }

    // Wait for semaphore for parent to drop to zero
    // N.B. this child may receive a KILL signal (kill -9)
    int sv{0};
    int icount{0};
    errno = 0;
    do
    {
        // Wait up to 5s while semaphore has a value of 0
        if (0!=(sv=ImageStreamIO_semvalue(&image,ISEM_CHILD_TO_PARENT)))
        {
            usleep(50000);  // Sleep for 100ms
        }
    } while (++icount<100 && sv && !errno);   // Wait at least 5s

    if (sv)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to see semaphore for parent drop to 0{"
        << (errno ? strerror(errno) : "unknown error")
        << "}"
        << std::endl;
        exit(7);
    }

    // Close and destroy shmim file
    if (IMAGESTREAMIO_SUCCESS != ImageStreamIO_destroyIm(&image))
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_shmim:  failed to destroy shmim{"
        << (errno ? strerror(errno) : "unknown error")
        << "}"
        << std::endl;
        exit(8);
    }

    exit(0);
}

// - ISIO_CLEANUP child sequence that works with process-local memory
void
ISIO_CLEANUP::run_child_sequence_local()
{
    // Create non-shared, process-local IMAGE memory data entity
    // - location is -9, which would cause an error for a shared shmim
    // - shared is 0
    // - NBsem is -99, which would cause an error for a shared shmim
    // - CBsize is -999, which would cause an error for a shared shmim
    uint32_t dims[2] = {2,3};
    errno_t createIm =
    ImageStreamIO_createIm_gpu(&image, "process-local/shmim"
                              , (sizeof dims) / (sizeof *dims)
                              , dims
                              , _DATATYPE_FLOAT
                              , -9          // Location
                              , 0           // Non-shared, process-local
                              , -99         // NBsem, ignored for local
                              , 10          // Keywords to be malloced
                              , MATH_DATA   // Data structure
                              , -999        // CBsize, ignored for local
                              );
    if (IMAGESTREAMIO_SUCCESS != createIm)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_local:  failed createIm{"
        << std::to_string(createIm)
        << "}"
        << std::endl;
        exit(1);
    }

    // ImageStreamIO_createIm_gpu will have malloc'ed space for keywords
    if (!image.kw)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_local:  keyword pointer is NULL"
        << std::endl;
        exit(2);
    }

    // ImageStreamIO_createIm_gpu will have calloc'ed space for data
    if (!image.array.raw)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_local:  data pointer is NULL"
        << std::endl;
        exit(3);
    }

    // ImageStreamIO_createIm_gpu will have malloc'ed space for metadata
    if (!image.md)
    {
        std::cerr
        << "ISIO_CLEANUP::run_child_sequence_local:  md pointer is NULL"
        << std::endl;
        exit(4);
    }

    // Calling free(...) on those pointers should not crash this child
    free(image.kw);
    free(image.array.raw);
    free(image.md);

    exit(0);
}

// /////////////////////////////////////////////////////////////////////
// ISIO_CLEANUP class utilities
// /////////////////////////////////////////////////////////////////////
int
ISIO_CLEANUP::find_semfiles(std::vector<std::string>& vsemfiles)
{
    // Put filepaths of named semaphore files under /dev/shm/ in vector

    vsemfiles.clear();
    DIR* dirshm = opendir("/dev/shm");
    if (!dirshm) { return errno ? errno : -1; }

    struct dirent* de;

    errno = 0;
    while ((de=readdir(dirshm)) && !errno)
    {
        // Add prefix=matching semaphore filenames to the vector
        std::string strel = std::string(de->d_name);
        std::size_t npos = strel.find(sem_prefix);
        if (npos==0) { vsemfiles.push_back("/dev/shm/" + strel); }
    }
    int save_errno{errno};
    closedir(dirshm);
    errno = save_errno;
    return errno;
}
#endif//u_IMAGESTREAMIO_CLEANUPTEST_HPP__
