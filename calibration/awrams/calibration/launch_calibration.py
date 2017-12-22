import os
import sys

def run_from_pickle(pklfile):
    '''
    Launch an MPI calibration job from the specified picklefile (usually built from cluster.build_pickle_from_spec)
    '''
    import pickle
    cspec = pickle.load(open(pklfile,'rb'))
    n_workers = cspec['n_workers']

    call_str = 'mpirun -x TMPDIR=/dev/shm/ -n 1 --map-by node  python3 -m awrams.calibration.server {pklfile} :\
     -x TMPDIR=/dev/shm/ -n {n_workers} python3 -m awrams.calibration.node'.format(**locals())


    from subprocess import Popen,PIPE,STDOUT,signal
    import sys
    proc = Popen(call_str.split(),stdout=PIPE,stderr=STDOUT)

    cur_out = ' '

    try:

        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        while cur_out != b'':
            cur_out = proc.stdout.read1(32768)
            sys.stdout.write(cur_out.decode())
            sys.stdout.flush()

    return_code = proc.wait()    


def get_nodelist():
    return list(set([n.strip() for n in open(os.environ['PBS_NODEFILE']).readlines()]))

def build_nodestring(nodelist):
    return ''.join([n+',' for n in nodelist])[:-1]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch a clustered job')
    parser.add_argument('pickle_file', type=str,
                        help='filename of pickled cal_spec')

    args = parser.parse_args()

    pklfile = args.pickle_file

    run_from_pickle(pklfile)
