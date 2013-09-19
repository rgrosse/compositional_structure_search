import glob
import os
import re
import smtplib
import socket
import subprocess
import sys

import config

def _status_path(key):
    return os.path.join(config.JOBS_PATH, key)

def _status_file(key, host=None):
    if host is not None:
        return os.path.join(_status_path(key), 'status-%s.txt' % host)
    else:
        return os.path.join(_status_path(key), 'status.txt')

def _run_job(script, key, args):
    if key != 'None':
        outstr = open(_status_file(key, socket.gethostname()), 'a')
        print >> outstr, 'running:', args
        outstr.close()
        
    ret = subprocess.call('python %s %s' % (script, args), shell=True)

    if key != 'None':
        outstr = open(_status_file(key, socket.gethostname()), 'a')
        if ret == 0:
            print >> outstr, 'finished:', args
        else:
            print >> outstr, 'failed:', args
        outstr.close()

def _executable_exists(command):
    # taken from stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ['PATH'].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, command)
        if is_exe(exe_file):
            return True

    return False

def _remove_status_files(key):
    fnames = os.listdir(_status_path(key))
    for fname in fnames:
        if re.match(r'status-.*.txt', fname):
            full_path = os.path.join(_status_path(key), fname)
            os.remove(full_path)

def run_command(command, jobs, machines=None, chdir=None):
    args = ['parallel']
    if machines is not None:
        for m in machines:
            args += ['--sshlogin', m]

    if chdir is not None:
        command = 'cd %s; %s' % (chdir, command)
    args += [command]

    p = subprocess.Popen(args, shell=False, stdin=subprocess.PIPE)
    p.communicate('\n'.join(jobs))

def run(script, jobs, machines=None, key=None, email=False, rm_status=True):
    if not _executable_exists('parallel'):
        raise RuntimeError('GNU Parallel executable not found.')
    if not hasattr(config, 'JOBS_PATH'):
        raise RuntimeError('Need to specify JOBS_PATH in config.py')
    if not os.path.exists(config.JOBS_PATH):
        raise RuntimeError('Path chosen for config.JOBS_PATH does not exist: %s' % config.JOBS_PATH)
    
    if key is not None:
        if not os.path.exists(_status_path(key)):
            os.mkdir(_status_path(key))
            
        outstr = open(_status_file(key), 'w')
        for job in jobs:
            print >> outstr, 'queued:', job
        outstr.close()

        if rm_status:
            _remove_status_files(key)
        
    command = 'python parallel.py %s %s' % (key, script)
    run_command(command, jobs, machines=machines, chdir=os.getcwd())

    if email:
        if key is not None:
            subject = '%s jobs finished' % key
            p = subprocess.Popen(['check_status', key], stdout=subprocess.PIPE)
            body, _ = p.communicate()
        else:
            subject = 'jobs finished'
            body = ''

        msg = '\r\n'.join(['From: %s' % config.EMAIL,
                           'To: %s' % config.EMAIL,
                           'Subject: %s' % subject,
                           '',
                           body])
        
        s = smtplib.SMTP('localhost')
        s.sendmail(config.EMAIL, [config.EMAIL], msg)
        s.quit()

def isint(p):
    try:
        int(p)
        return True
    except:
        return False

def parse_machines(s, njobs):
    if s is None:
        return s
    parts = s.split(',')
    return ['%d/%s' % (njobs, p) for p in parts]

def list_jobs(key, status_val):
    status_files = [os.path.join(_status_path(key), 'status.txt')]
    status_files += glob.glob('%s/status-*.txt' % _status_path(key))

    status = {}
    for fname in status_files:
        for line_ in open(fname).readlines():
            line = line_.strip()
            sv, args = line.split(':')
            args = args.strip()
            status[args] = sv

    return [k for k, v in status.items() if v == status_val]


if __name__ == '__main__':
    assert len(sys.argv) == 4
    key = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3]
    _run_job(script, key, args)
