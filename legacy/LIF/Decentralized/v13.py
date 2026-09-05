from p2pnetwork.node import Node
from p2pnetwork.nodeconnection import NodeConnection
import signal
import sys
import time

class NeuronNodeConnection (NodeConnection):
    # Python class constructor
    def __init__(self, main_node, sock, id, host, port):
        super().__init__(main_node, sock, id, host, port)

class NeronNode (Node):
    # Python class constructor
    def __init__(self, host, port, id=None, callback=None, max_connections=0):
        super(NeronNode, self).__init__(host, port, id, callback, max_connections)

    def outbound_node_connected(self, connected_node):
        print("outbound_node_connected: " + connected_node.id)
        
    def inbound_node_connected(self, connected_node):
        print("inbound_node_connected: " + connected_node.id)

    def inbound_node_disconnected(self, connected_node):
        print("inbound_node_disconnected: " + connected_node.id)

    def outbound_node_disconnected(self, connected_node):
        print("outbound_node_disconnected: " + connected_node.id)

    def node_message(self, connected_node, data):
        
        print("node_message from " + connected_node.id + ": " + str(data))
        
    def node_disconnect_with_outbound_node(self, connected_node):
        print("node wants to disconnect with oher outbound node: " + connected_node.id)
        
    def node_request_to_stop(self):
        print("node is requested to stop!")

    # OPTIONAL
    # If you need to override the NodeConection as well, you need to
    # override this method! In this method, you can initiate
    # you own NodeConnection class.
    def create_new_connection(self, connection, id, host, port):
        return NeuronNodeConnection(self, connection, id, host, port)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    node.stop()
    time.sleep(2)
    sys.exit(0)

inbound = input("Your Addr:Port >>> ").split(":")
inbound_ip = str(inbound[0])
inbound_port = int(inbound[1])

id = int(input("ID >>> "))

node = NeronNode(inbound_ip, inbound_port, id)
node.start()

outbound = input("Connect >>> ").split(":")
outbound_addr = str(outbound[0])
outbound_port = int(outbound[1])

node.connect_with_node(outbound_addr, outbound_port)
print(node.nodes_inbound)
print(node.nodes_outbound)
node.send_to_nodes("Hello!")

