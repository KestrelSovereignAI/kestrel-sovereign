"""Partition collected pytest nodes and verify complete, disjoint CI shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def shard_for_node(nodeid: str, count: int) -> int:
    if not isinstance(nodeid, str) or not nodeid or type(count) is not int or count < 1:
        raise ValueError('A node ID and positive shard count are required')
    digest = hashlib.sha256(nodeid.encode('utf-8', 'surrogatepass')).digest()
    return int.from_bytes(digest, 'big') % count + 1


def _same_node_membership(first, second):
    return len(first) == len(second) and set(first) == set(second)


def _valid_worker(worker) -> bool:
    return isinstance(worker, str) and bool(worker) and all(c.isalnum() or c in '_-' for c in worker)


def pytest_addoption(parser):
    group = parser.getgroup('unit sharding')
    group.addoption('--unit-shard', type=int, default=None)
    group.addoption('--unit-shard-count', type=int, default=None)
    group.addoption('--unit-shard-manifests', default=None)


def pytest_collection_modifyitems(config, items):
    index = config.getoption('unit_shard')
    count = config.getoption('unit_shard_count')
    directory = config.getoption('unit_shard_manifests')
    if index is None and count is None and directory is None:
        yield
        return
    import pytest

    if (type(index) is not int or type(count) is not int
            or not 1 <= index <= count or not directory):
        raise pytest.UsageError('Supply a valid unit shard, count and manifest directory together')
    original = [item.nodeid for item in items]
    if len(original) != len(set(original)):
        raise pytest.UsageError('Unit shard collection contains duplicate node IDs')
    selected, deselected = [], []
    for item in items:
        (selected if shard_for_node(item.nodeid, count) == index else deselected).append(item)
    worker = getattr(config, 'workerinput', {}).get('workerid', 'controller')
    if not _valid_worker(worker):
        raise pytest.UsageError('Invalid unit shard worker identity')
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
    yield
    if not _same_node_membership([item.nodeid for item in items], [item.nodeid for item in selected]):
        raise pytest.UsageError('Unit shard final collection changed after assignment')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'shard-{index}-{worker}.json'
    with target.open('x', encoding='utf-8') as output:
        json.dump(dict(format_version=1, shard=index, shard_count=count, worker=worker,
            all_nodeids=original, selected_nodeids=[item.nodeid for item in items]), output)


def verify_manifests(directory: Path, count: int) -> dict:
    if type(count) is not int or count < 1:
        raise ValueError('A positive shard count is required')
    files = sorted(directory.glob('*.json'))
    if not files:
        raise ValueError('No unit shard collection manifests were produced')
    groups = {}
    full = None
    workers = set()
    for path in files:
        report = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report, dict) or set(report) != {'format_version', 'shard', 'shard_count', 'worker', 'all_nodeids', 'selected_nodeids'}:
            raise ValueError(f'Invalid unit shard manifest fields: {path.name}')
        index = report['shard']
        if (type(report['format_version']) is not int or report['format_version'] != 1
                or type(report['shard_count']) is not int or report['shard_count'] != count
                or type(index) is not int or not 1 <= index <= count
                or not _valid_worker(report['worker'])):
            raise ValueError(f'Unexpected unit shard assignment: {path.name}')
        original, selected = report['all_nodeids'], report['selected_nodeids']
        if any(not isinstance(nodes, list) or any(not isinstance(n, str) or not n for n in nodes)
               or len(nodes) != len(set(nodes)) for nodes in (original, selected)):
            raise ValueError(f'Invalid or duplicate unit shard node IDs: {path.name}')
        if not original:
            raise ValueError('The full unit collection must not be empty')
        if full is None:
            full = original
        if original != full:
            raise ValueError('Unit shard workers collected different full suites')
        worker_key = (index, report['worker'])
        if worker_key in workers:
            raise ValueError('Duplicate unit shard worker manifest')
        workers.add(worker_key)
        expected = [node for node in original if shard_for_node(node, count) == index]
        if not _same_node_membership(selected, expected):
            raise ValueError('Unit shard selection omitted or misassigned collected tests')
        if index in groups and groups[index] != selected:
            raise ValueError("Unit shard workers collected different final execution orders")
        groups[index] = selected
    if set(groups) != set(range(1, count + 1)):
        raise ValueError('Missing unit shard collection manifests')
    assigned = [node for index in sorted(groups) for node in groups[index]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(full):
        raise ValueError('Unit shards must cover every collected test exactly once')
    return dict(complete=True, shard_count=count, worker_manifests=len(files),
        collected_tests=len(full), assigned_tests=len(assigned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_manifests(args.verify, args.count)))


if __name__ == '__main__':
    main()
else:
    import pytest

    pytest_collection_modifyitems = pytest.hookimpl(wrapper=True, tryfirst=True)(pytest_collection_modifyitems)
