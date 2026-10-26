# Browser substrate only. The governed session driver is a separate layer.
FROM mcr.microsoft.com/playwright/python:v1.63.0-noble@sha256:72bd171a9ffc2b4b59532aaa6210e21014d07093120dc25528870c0b840da1f0

LABEL io.kestrel.visual-computer.worker-contract="sdk-0.43-workspace-v1"
COPY requirements.txt /opt/kestrel-browser/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /opt/kestrel-browser/requirements.txt
COPY sandbox_probe.py sandbox_checks.py controller.py worker.py wire.py rpc_client.py workspace.py /opt/kestrel-browser/
USER pwuser
