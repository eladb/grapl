from multiprocessing.connection import Connection

from pydgraph import DgraphClient, DgraphClientStub

from graph import Process, File, Not, batch_queries
from analyzerlib import analyze_by_node_key, ExecutionComplete, NodeRef, ExecutionHit, Subgraph

# Look for processes with svchost.exe in their name with non services.exe parents
def signature_graph(node_key) -> str:
    svchost = Process() \
        .with_image_name(contains="svchost.exe") \
        .with_node_key(eq=node_key)

    bin_file = File() \
        .with_path(eq=Not("C:\\\\Windows\\\\System32\\\\svchost.exe"))

    return svchost.with_bin_file(bin_file).to_query()


def _analyzer(client: DgraphClient, graph: Subgraph, sender: Connection):
    hits = analyze_by_node_key(client, graph, signature_graph)

    for hit in hits:
        sender.send(ExecutionHit.from_process_file('suspicious-svchost-bin_file', hit))

    sender.send(ExecutionComplete())


def analyzer(graph: Subgraph, sender: Connection):
    try:
        print('analyzing')
        client = DgraphClient(DgraphClientStub('db.mastergraph:9080'))
        _analyzer(client, graph, sender)
    except Exception as e:
        print('analyzer failed: {}'.format(e))
        sender.send(None)
