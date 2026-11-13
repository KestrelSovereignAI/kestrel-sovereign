"""Run Core's operator forms with published MCP in its original Docker namespace."""

import argparse
from contextlib import closing
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tomllib
import xml.etree.ElementTree as ET
import zipfile

def tracked_files(repository):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repository).decode().split('\0')
    for name in filter(None, names):
        path = repository / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Expected a regular tracked build input: {name}')
        yield name, path.read_bytes()


def wheel_inventory(wheel, repository):
    """Refuse a candidate whose packaged Core bytes differ from this checkout."""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist()
                 if name.startswith('kestrel_sovereign/') and not name.endswith('/')]
        if not names or len(names) != len(set(names)):
            raise RuntimeError('Core wheel has an empty or duplicate runtime inventory')
        result = {}
        for name in names:
            body = archive.read(name)
            if body != (repository / name).read_bytes():
                raise RuntimeError(f'Core wheel differs from source: {name}')
            result[name] = hashlib.sha256(body).hexdigest()
        return result


def runtime_install_command(python, mcp_version, core_wheel):
    """Install the exact runtime pair and resolve its declared dependencies."""
    return ['uv', 'pip', 'install', '--python', str(python),
            '--index-url', 'https://pypi.org/simple', '--reinstall',
            f'kestrel-feature-mcp=={mcp_version}', str(core_wheel)]


def build_input_image(client, owner, core, mcp, output):
    # The MCP project owns one browser/runtime build and namespace owner.
    # Its checkout supplies test support; runtime imports below must come from
    # the public MCP wheel and the candidate Core wheel in site-packages.
    base = owner.build_image(client, mcp, output)
    wheels = output / 'wheels'
    env = {key: os.environ[key] for key in ('PATH', 'LANG') if key in os.environ}
    env.update(HOME=str(output / 'build-home'), UV_CACHE_DIR=str(output / 'uv-cache'),
               UV_NO_CONFIG='1', PYTHON_DOTENV_DISABLED='1', KESTREL_SKIP_DOTENV='1')
    subprocess.run(['uv', 'build', '--wheel', '--no-sources', '--out-dir', str(wheels)],
                   cwd=core, env=env, check=True)
    wheel, = wheels.glob('*.whl')
    core_files = wheel_inventory(wheel, core)
    mcp_files = {name: hashlib.sha256(body).hexdigest() for name, body in tracked_files(mcp)
                 if name.startswith('kestrel_feature_mcp/') or name == 'kestrel_mcp_apps_sandbox.py'}
    mcp_version = tomllib.loads((mcp / 'pyproject.toml').read_text())['project']['version']
    core_version = tomllib.loads((core / 'pyproject.toml').read_text())['project']['version']
    manifest = dict(core=dict(version=core_version, files=core_files),
                    mcp=dict(version=mcp_version, files=mcp_files))
    verify = '''import hashlib,importlib,importlib.metadata as metadata,json,pathlib,sys
d=json.load(open('/owned/runtime.json'))
for label,package,distribution in [('core','kestrel_sovereign','kestrel-sovereign'),
                                  ('mcp','kestrel_feature_mcp','kestrel-feature-mcp')]:
 module=importlib.import_module(package)
 assert pathlib.Path(module.__file__).is_relative_to(pathlib.Path(sys.prefix)), (label,module.__file__)
 dist=metadata.distribution(distribution)
 assert dist.version==d[label]['version'], (label,dist.version)
 if label=='mcp': assert dist.read_text('direct_url.json') is None, 'MCP must be installed from PyPI'
 for name,digest in d[label]['files'].items():
  assert hashlib.sha256(pathlib.Path(dist.locate_file(name)).read_bytes()).hexdigest()==digest, (label,name)
'''
    run = '''import runpy,pytest,sys
sys.path.insert(0,'/core-tests')
runpy.run_path('/owned/verify.py')
try:
 result=pytest.main(['tests/integration/test_mcp_input_host_browser.py','-q',
  '-p','pytest_asyncio.plugin','-p','pytest_timeout','-p','kestrel_sovereign.testing.fixtures',
  '--confcutdir=tests/integration','-o','addopts=','-o','asyncio_mode=auto','--timeout=60',
  '--junitxml=/input-browser.xml','--basetemp=/owned-input-browser-tests'])
finally: runpy.run_path('/owned/verify.py')
raise SystemExit(result)
'''
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode='w') as archive:
        def add(name, body):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(body), 0o644
            archive.addfile(item, io.BytesIO(body))
        for name, body in tracked_files(core):
            if name.startswith('tests/') or name == 'pyproject.toml':
                add('core-tests/' + name, body)
        add('owned/' + wheel.name, wheel.read_bytes())
        add('owned/runtime.json', json.dumps(manifest).encode())
        add('owned/verify.py', verify.encode())
        add('owned/run.py', run.encode())
        add('.dockerignore', b'')
        install = runtime_install_command('/work/.venv/bin/python', mcp_version,
                                          '/owned/' + wheel.name)
        add('Dockerfile', f'''FROM {base}
COPY core-tests/ /core-tests/
COPY owned/ /owned/
RUN {shlex.join(install)}
RUN uv pip check --python /work/.venv/bin/python
ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 MCP_TEST_BROWSER_SUPPORT=/work
WORKDIR /core-tests
RUN /work/.venv/bin/python -I -B /owned/verify.py
CMD ["/work/.venv/bin/python", "-I", "-B", "/owned/run.py"]
'''.encode())
    context.seek(0)
    image_id = None
    with (output / 'input-build.jsonl').open('w') as log:
        for row in client.api.build(fileobj=context, custom_context=True,
                                    rm=True, decode=True, use_config_proxy=False):
            log.write(json.dumps(row) + '\n')
            log.flush()
            if 'error' in row:
                raise RuntimeError(row['error'])
            if row.get('aux', {}).get('ID'):
                image_id = row['aux']['ID']
    if not image_id:
        raise RuntimeError('Input browser build returned no immutable image ID')
    owner.save_receipt(output / 'input-build.json', dict(image_id=image_id, runtime=manifest,
        core_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), base_image=base))
    return image_id


def run_acceptance(client, owner, image_id, output):
    with owner.original_container(client, image_id, output / 'container.json') as (cid, receipt):
        client.api.start(cid)
        receipt['test_wait'] = client.api.wait(cid, timeout=300)
        (output / 'browser.log').write_bytes(client.api.logs(cid))
        xml = owner.container_file(client, cid, '/input-browser.xml')
        (output / 'input-browser.xml').write_bytes(xml)
        cases = list(ET.fromstring(xml).iter('testcase'))
        counts = {kind: sum(case.find(kind) is not None for case in cases)
                  for kind in ('failure', 'error', 'skipped')}
        receipt.update(tests=len(cases), results=counts)
        owner.save_receipt(output / 'container.json', receipt)
        if receipt['test_wait']['StatusCode'] or not cases or any(counts.values()):
            raise RuntimeError(f'Native input browser acceptance failed: {output / "browser.log"}')


def main():
    import docker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mcp-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    mcp = args.mcp_source.resolve(strict=True)
    core = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('mcp_browser_owner', mcp / 'tests/browser/run_native.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    with closing(docker.from_env(timeout=30)) as client:
        if args.cleanup:
            path = args.output / 'container.json'
            if path.exists():
                receipt = json.loads(path.read_text())
                owner.retire_container(client, receipt['container_id'], receipt, path)
            return
        args.output.mkdir(parents=True, exist_ok=False)
        image_id = build_input_image(client, owner, core, mcp, args.output)
        run_acceptance(client, owner, image_id, args.output)


if __name__ == '__main__':
    main()
