#include "sysutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/file.h>
#include <string.h>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

#define BT_DEPTH 100

static void _critical_stop_handler(int signum, siginfo_t *info, void *ucontext)
{
    void *callstack[BT_DEPTH];
    int i, frames = 0;
    char **strs;
    void *caller_address = NULL;
    ucontext_t *uc = ucontext;

    signal(signum, SIG_DFL); /* Avoid infinite loops */
    signal(SIGABRT, SIG_DFL);
    fflush(stdout);
    fprintf(stderr, "\n\nCritical error: received signal %s. Unexpected exit.\n", strsignal(signum));
    fprintf(stderr, "Trying to get the backtrace (max. depth %d)...\n", BT_DEPTH);

    frames = backtrace(callstack, BT_DEPTH);

    if(uc != NULL)
    {
#ifdef __APPLE__
    caller_address = (void*) uc->uc_mcontext->__ss.__rip;
#endif
    }

    if(frames > 3 && caller_address != NULL)
        callstack[2] = caller_address;

    strs = backtrace_symbols(callstack, frames);

    for (i = 0; i < frames; ++i)
        fprintf(stderr, "%s\n", strs[i]);

    free(strs);

    abort();
}

int install_stop_handlers()
{
    int retval = 0;

    if (signal(SIGSEGV, (sig_t)_critical_stop_handler) == SIG_ERR)
    {
        perror("signal: SIGSEGV");
        retval = -1;
    }

    if (signal(SIGILL, (sig_t)_critical_stop_handler) == SIG_ERR)
    {
        perror("signal: SIGILL");
        retval = -1;
    }

    if (signal(SIGBUS, (sig_t)_critical_stop_handler) == SIG_ERR)
    {
        perror("signal: SIGBUS");
        retval = -1;
    }

    if (signal(SIGABRT, (sig_t)_critical_stop_handler) == SIG_ERR)
    {
        perror("signal: SIGABRT");
        retval = -1;
    }

    if (signal(SIGFPE, (sig_t)_critical_stop_handler) == SIG_ERR)
    {
        perror("signal: SIGFPE");
        retval = -1;
    }

    return retval;
}
