import os
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

def run(script, jobs, machines=None, key=None, email=False):
    if key is not None:
        if not os.path.exists(_status_path(key)):
            os.mkdir(_status_path(key))
            
        outstr = open(_status_file(key), 'w')
        for job in jobs:
            print >> outstr, 'queued:', job
        outstr.close()

        subprocess.call('cd %s; rm status-*.txt' % _status_path(key), shell=True)
        
    command = 'python utils/parallel.py %s %s' % (key, script)
    run_command(command, jobs, machines=machines, chdir=config.CODE_PATH)

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

if __name__ == '__main__':
    assert len(sys.argv) == 4
    key = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3]
    _run_job(script, key, args)
