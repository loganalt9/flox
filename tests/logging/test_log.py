from flox.nn.logger import BaseLogger, CSVLogger, TensorBoardLogger
from datetime import datetime
from flox.flock import Flock, FlockNode
from flox import federated_fit
from flox.data.utils import federated_split
import os

def test_csv_protocol():
    logger = CSVLogger()
    assert isinstance(logger, CSVLogger)
    assert isinstance(logger, BaseLogger)

def test_tensorboard_protocol():
    logger = TensorBoardLogger()
    assert isinstance(logger, TensorBoardLogger)
    assert isinstance(logger, BaseLogger)

def test_csv_output_format():
    logger = CSVLogger()
    expect_records = []

    dt1 = datetime.now()
    logger.log('train/loss', 2.3, 'node 1', 0, dt1)
    expect_records.append({'name': 'train/loss', 'value': 2.3, 'nodeid': 'node 1', 'epoch': 0, 'datetime': dt1})
    dt2 = datetime.now()
    logger.log('train/loss', 2.4, 'node 2', 0, dt2)
    expect_records.append({'name': 'train/loss', 'value': 2.4, 'nodeid': 'node 2', 'epoch': 0, 'datetime': dt2})
    dt3 = datetime.now()
    logger.log('train/loss', 1.4, 'node 1', 1, dt3)
    expect_records.append({'name': 'train/loss', 'value': 1.4, 'nodeid': 'node 1', 'epoch': 1, 'datetime': dt3})
    dt4 = datetime.now()
    logger.log('train/loss', 1.2, 'node 2', 1, dt4)
    expect_records.append({'name': 'train/loss', 'value': 1.2, 'nodeid': 'node 2', 'epoch': 1, 'datetime': dt4})


    actual: str = logger.export(None)
    expected: str = f"name,value,nodeid,epoch,datetime\ntrain/loss,2.3,node 1,0,{dt1}\ntrain/loss,2.4,node 2,0,{dt2}\ntrain/loss,1.4,node 1,1,{dt3}\ntrain/loss,1.2,node 2,1,{dt4}"

    assert expected.lower().splitlines() == actual.lower().splitlines()
    assert expect_records == logger.records
"""
def test_csv_flox():
    topo = {
        "Mycoordinator": {
            "kind": "leader",
            "children": ["Aggr","Worker3"],
            "globus_compute_endpoint": None,
            "proxystore_endpoint": "<UUID>"
        },
        "Aggr": {
            "kind": "aggregator",
            "children": ["Worker1", "Worker2"],
            "globus_compute_endpoint": "<UUID>",
            "proxystore_endpoint": "<UUID>",
        },
        "Worker1": {
            "kind": "worker",
            "children": [],
            "globus_compute_endpoint": "<UUID>",
            "proxystore_endpoint": "<UUID>"
        },
        "Worker2": {
            "kind": "worker",
            "children": [],
            "globus_compute_endpoint": "<UUID>",
            "proxystore_endpoint": "<UUID>"
        },
        "Worker3": {
            "kind": "worker",
            "children": [],
            "globus_compute_endpoint": "<UUID>",
            "proxystore_endpoint": "<UUID>"
        }
    }

    flock = Flock.from_yaml('./flox/tests/logging/examples/three-level.yaml')

    fed_data = federated_split(
        topo, flock, num_classes=10,
        samples_alpha=1.0, labels_alpha=1.0
    )

    trained_model, results = federated_fit(
        topo, MyModule(), fed_data,
        strategy='fedavg',
        where='local',
        kind='sync'
    )

    assert True
""" 

def test_tensorboard():
    node_worker1 = FlockNode(idx='node1', kind='worker')
    node_worker2 = FlockNode(idx='node2', kind='worker')

    exp_rec1 = []
    exp_rec2 = []

    logger_worker1 = TensorBoardLogger(node_worker1)
    logger_worker2 = TensorBoardLogger(node_worker2)

    import numpy as np
    for step in range(1,101):

        rand1 = np.random.random()
        dt1 = datetime.now()
        logger_worker1.log('Train/Loss', rand1 / step, 'node 1', step, dt1)
        exp_rec1.append({'name': 'Train/Loss', "value": rand1/step, 'nodeid': 'node 1', 'epoch': step, 'datetime': dt1})

        rand2 = np.random.random()
        dt2 = datetime.now()
        logger_worker2.log('Train/Loss', rand2 / step, 'node 2', step, dt2)
        exp_rec2.append({'name': 'Train/Loss', "value": rand2/step, 'nodeid': 'node 2', 'epoch': step, 'datetime': dt2})
    
    assert exp_rec1 == logger_worker1.records
    assert exp_rec2 == logger_worker2.records

    check_dir_made: bool = os.path.exists('./runs')
    assert check_dir_made

def test_switch():
    logger_csv = CSVLogger()
    logger_tb = TensorBoardLogger()

    assert isinstance(logger_csv, BaseLogger) and isinstance(logger_tb, BaseLogger)

    expected = []

    dt1 = datetime.now()
    logger_csv.log('train/loss', 2.3, 'node 1', 0, dt1)
    expected.append({'name': 'train/loss', "value": 2.3, 'nodeid': 'node 1', 'epoch': 0, 'datetime': dt1})

    new_logger: TensorBoardLogger = logger_csv

    assert expected == new_logger.records