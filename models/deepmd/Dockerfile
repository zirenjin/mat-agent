FROM deepmodeling/deepmd-kit:3.1.2_cuda129

RUN mkdir -p /opt/mat-agent-tools
COPY tools/ase_conventions.py /opt/mat-agent-tools/
COPY models/deepmd/internal /opt/internal

RUN pip3 install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    "ase>=3.22" \
    "huggingface_hub>=0.20" \
    "dpdata>=0.2.18"

ENV PYTHONPATH=/opt/mat-agent-tools:/opt/internal
WORKDIR /work
ENTRYPOINT []
