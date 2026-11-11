# Imagem do BioHub: executa a CLI (biohub.py) com as dependências de visualização.
#
# Build:  docker build -t biohub .
# Uso:    docker run --rm -v "$PWD:/data" biohub fetchpdb 1TUP --chains A --protein-only -o 1TUP_clean.pdb
#
# Os arquivos de entrada e saída ficam no diretório montado em /data.
# PyMOL, PDB2PQR e APBS não estão incluídos (veja "Uso com Docker" no README).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY biohub.py biohub_viz.py ./

RUN useradd --create-home --uid 1000 biohub
USER biohub

# Pré-gera o cache de fontes do matplotlib para acelerar o primeiro gráfico.
RUN python -c "import matplotlib.font_manager"

WORKDIR /data

ENTRYPOINT ["python", "/app/biohub.py"]
