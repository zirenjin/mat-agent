FROM mat-agent/base:latest

# Install torch first from pytorch.org (CUDA version, reliable ABI)
# Then mace-torch from tsinghua (fast)
RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu126 \
        "torch==2.6.0" \
    && python3 -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

RUN pip3 install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        "mace-torch>=0.3.10" \
    && python3 -c "from mace.calculators import mace_mp; print('mace_mp OK')" \
    && mace_eval_configs --help > /dev/null 2>&1 \
    && echo "mace_eval_configs OK"


COPY requirements-data.txt /tmp/requirements-data.txt
RUN pip3 install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        -r /tmp/requirements-data.txt \
    && rm /tmp/requirements-data.txt

COPY models/mace/internal /opt/internal
ENV PYTHONPATH=/opt/mat-agent-tools:/opt/internal
WORKDIR /work
ENTRYPOINT []
