'''
Simple runner for single coefficient runoff model
'''


class RunoffRunner:
	def __init__(self):
		pass

	def run_from_mapping(self,mapping,timesteps,cells):
		outputs = dict(final_states=dict())
		outputs['q'] = mapping['c'] * mapping['precip']
		return outputs