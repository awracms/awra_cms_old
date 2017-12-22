from nose.tools import nottest, with_setup
import os

def clear_files():
    import glob
    files = glob.glob(os.path.join(os.path.dirname(__file__),'*.nc'))
    for f in files:
        os.remove(f)

def setup():
    clear_files()
    from awrams.models.awral import model
    
    model.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    global awral

    awral = model.AWRALModel()


def tear_down():
    clear_files()

def climate_mod(input_map):
    input_map['precip_f'].args.pattern = "rain*"
    input_map['tmin_f'].args.pattern = "temp_min*"
    input_map['tmax_f'].args.pattern = "temp_max*"
    input_map['solar_f'].args.pattern = "solar*"

# @nottest
@with_setup(setup,tear_down)
def test_SplitFileWriterNode():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    extent = extents.get_default_extent()


    from awrams.utils.nodegraph import nodes
    from awrams.simulation.ondemand import OnDemandSimulator

    input_map = awral.get_default_mapping()
    climate_mod(input_map)

    from awrams.utils.nodegraph import nodes
    from awrams.utils.metatypes import ObjectDict

    # output_path = './'
    output_map = awral.get_output_mapping()
    output_map['qtot_save'] = nodes.write_to_annual_ncfile('./','qtot')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    period = dt.dates('2010-2011')
    extent = extent.ioffset[200,200:202]
    r = runner.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_flatfm():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    input_map = awral.get_default_mapping()
    climate_mod(input_map)
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_ncfile(os.path.dirname(__file__),'s0')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2012')
    extent = e_all.ioffset[200:202,200:202]
    r = runner.run(period,extent)

    clear_files()

    output_map['s0_save'] = nodes.write_to_ncfile(os.path.dirname(__file__),'s0',mode='r+')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2013-2014')
    extent = e_all.ioffset[200:202,200:202]
    r = runner.run(period,extent)

    clear_files()

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2015')
    extent = e_all.ioffset[202,202]
    r = runner.run(period,extent)

    clear_files()

# @nottest
@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_A():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200:202,200:202]

    input_map = awral.get_default_mapping()
    climate_mod(input_map)
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    r = runner.run(period,extent)

@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_B():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator

    input_map = awral.get_default_mapping()
    climate_mod(input_map)
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0',mode='w')

    runner = OnDemandSimulator(awral,input_map,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[200:202,200:202]
    r = runner.run(period,extent)

    clear_files()

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2015')
    extent = e_all.ioffset[202,202]
    r = runner.run(period,extent)

@with_setup(setup,tear_down)
def test_output_graph_processing_splitfm_C():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    e_all = extents.get_default_extent()

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator


    print("RUNNER NEW: single cell ncf, multiple years")
    period = dt.dates('2010-2011')
    extent = e_all.ioffset[202,202]

    input_map = awral.get_default_mapping()
    climate_mod(input_map)
    output_map = awral.get_output_mapping()
    output_map['s0_save'] = nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map,omapping=output_map)
    r = runner.run(period,extent)

if __name__ == '__main__':
    # test_FileWriterNode()
    # test_SplitFileWriterNode()
    # test_output_graph_processing_flatfm()
    # test_output_graph_processing_splitfm()
    # test_output_graph()
    # test_OutputNode()
    pass
