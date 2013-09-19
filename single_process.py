import subprocess


def run(script, jobs):
    for job in jobs:
        subprocess.call(['python', script] + list(job))


