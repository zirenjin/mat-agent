FROM mat-agent/base:latest

RUN pip3 install --no-cache-dir     -i https://pypi.tuna.tsinghua.edu.cn/simple     --trusted-host pypi.tuna.tsinghua.edu.cn     fairchem-core>=2.20

COPY models/fairchem/_verify.py /tmp/_verify.py
RUN python3 /tmp/_verify.py && rm /tmp/_verify.py

COPY models/fairchem/internal /opt/internal
ENV PYTHONPATH=/opt/mat-agent-tools:/opt/internal
WORKDIR /work
ENTRYPOINT []
