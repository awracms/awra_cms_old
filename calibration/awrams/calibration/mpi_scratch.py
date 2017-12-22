
if __name__ == '__main__':

	import multiprocessing as mp
	mp.set_start_method('fork')

	from mpi4py import MPI
	import numpy as np

	comm = MPI.COMM_WORLD
	
	schema = ['qtot_nse','etot_nse']

	granks = [0,1,2,3,5]
	catch_layout = [0,2,4,1,3]

	ncatch = sum(catch_layout)

	catch_obj_sizes = np.array(catch_layout) * len(schema)
	catch_obj_offsets = [0] + list(np.cumsum(catch_obj_sizes)[:-1])

	catch_obj_sizes,catch_obj_offsets

	if comm.rank in granks:
		g = comm.Get_group()
		g = g.Incl(granks)
		print(g.Get_size(),g.Get_rank())
		icomm = comm.Create_group(g)

		rank = g.Get_rank()

		if rank == 0:
			recvbuf = np.zeros((ncatch,len(schema)))
			sendbuf = bytearray(0)
			r = icomm.Igatherv(sendbuf,[recvbuf,catch_obj_sizes,catch_obj_offsets,MPI.DOUBLE])
			r.Wait()
			print(schema[0],recvbuf.T[0])
			print(schema[1],recvbuf.T[1])
		else:
			sendbuf = np.zeros(len(schema)*catch_layout[rank])
			sendbuf[...] = rank * 10 + np.tile(np.arange(len(schema)),catch_layout[rank])

			icomm.Igatherv(sendbuf,[None,catch_obj_sizes,catch_obj_offsets,MPI.DOUBLE])

		#icomm.send('a',0)
		#icomm.recv()
