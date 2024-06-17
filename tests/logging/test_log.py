from flox.nn.logger import BaseLogger, CSVLogger, TensorBoardLogger

def test_csv_protocol():
    logger = CSVLogger()
    assert isinstance(logger, CSVLogger)
    assert isinstance(logger, BaseLogger)

def test_tensorboard_protocol():
    logger = TensorBoardLogger()
    assert isinstance(logger, TensorBoardLogger)
    assert isinstance(logger, BaseLogger)

def test_csv():
    logger = CSVLogger()
    expect_records = []
    logger.log('Node 1', 2.3, 0)
    expect_records.append({'name': 'Node 1', 'train/loss': 2.3, 'round': 0})
    logger.log('Node 2', 2.4, 0)
    expect_records.append({'name': 'Node 2', 'train/loss': 2.4, 'round': 0})
    logger.log('Node 1', 1.4, 1)
    expect_records.append({'name': 'Node 1', 'train/loss': 1.4, 'round': 1})
    logger.log('Node 2', 1.2, 1)
    expect_records.append({'name': 'Node 2', 'train/loss': 1.2, 'round': 1})


    actual: str = logger.export(None)
    expected: str = "name,train/loss,round\nNode 1,2.3,0\nNode 2,2.4,0\nNode 1,1.4,1\nNode 2,1.2,1"

    assert expected.lower().splitlines() == actual.lower().splitlines()
    assert expect_records == logger.records