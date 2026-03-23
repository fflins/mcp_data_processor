from mcp.server.fastmcp import FastMCP
import pandas as pd
import io
from data_proc import remover_duplicatas, remover_nulos, normalizar_coluna, renomear_colunas, remover_colunas, detectar_outliers, exportar_csv, codificar_categoricas



mcp = FastMCP("preprocessamento")

dataset = {}

@mcp.tool()
def tool_carregar_csv_local(id_dataset: str, caminho_arquivo: str) -> str:
    """
    Carrega um arquivo CSV do sistema de arquivos local do computador para a memória do servidor Pandas.
    Obrigatório usar esta ferramenta antes de tentar limpar ou inspecionar os dados.
    """
    try:
        df = pd.read_csv(caminho_arquivo)
        dataset[id_dataset] = df
        return f"Sucesso: Dataset '{id_dataset}' carregado do disco. Shape inicial: {df.shape}. Colunas: {list(df.columns)}"
    except FileNotFoundError:
        return f"Erro: O arquivo não foi encontrado no caminho especificado: '{caminho_arquivo}'."
    except Exception as e:
        return f"Erro ao ler o arquivo local: {str(e)}"

@mcp.tool()
def tool_inspecionar_dados(id_dataset: str) -> str:
    """
    Retorna as 5 primeiras linhas do dataset, além de um resumo estatístico das colunas e os tipos de dados contidos na memória.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado no contexto."
    
    df = dataset[id_dataset]
    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    
    return f"Resumo (Info):\n{info_buffer.getvalue()}\n\nAmostra (Head):\n{df.head().to_markdown()}"

@mcp.tool()
def tool_remover_duplicatas(id_dataset: str) -> str:
    """
    Remove todas as linhas idênticas (duplicadas) do dataset especificado.
    """
    if id_dataset not in dataset:
        return f"Erro: Dataset '{id_dataset}' não existe no contexto."
    
    df_atual = dataset[id_dataset]
    linhas_antes = len(df_atual)
    dataset[id_dataset] = remover_duplicatas(df_atual)
    linhas_removidas = linhas_antes - len(dataset[id_dataset])
    
    return f"Removidas {linhas_removidas} linhas duplicadas. Shape atual: {dataset[id_dataset].shape}"

@mcp.tool()
def tool_remover_nulos(id_dataset: str) -> str:
    """
    Remove todas as linhas que contêm valores nulos (NaN/Null) do dataset.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
        
    df_atual = dataset[id_dataset]
    linhas_antes = len(df_atual)
    dataset[id_dataset] = remover_nulos(df_atual)
    linhas_removidas = linhas_antes - len(dataset[id_dataset])
    
    return f"Drop de nulos concluído. {linhas_removidas} linhas descartadas. Novo shape: {dataset[id_dataset].shape}"

@mcp.tool()
def tool_renomear_colunas(id_dataset: str, mapeamento: dict) -> str:
    """
    Renomeia colunas do dataset. Recebe um dicionário com {'nome_atual': 'nome_novo'}.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
    dataset[id_dataset] = renomear_colunas(dataset[id_dataset], mapeamento)
    return f"Colunas renomeadas. Colunas atuais: {list(dataset[id_dataset].columns)}"

@mcp.tool()
def tool_remover_colunas(id_dataset: str, colunas: list) -> str:
    """
    Remove uma ou mais colunas do dataset.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
    dataset[id_dataset] = remover_colunas(dataset[id_dataset], colunas)
    return f"Colunas removidas. Colunas atuais: {list(dataset[id_dataset].columns)}"

@mcp.tool()
def tool_detectar_outliers(id_dataset: str) -> str:
    """
    Detecta outliers nas colunas numéricas usando o método IQR e retorna a contagem por coluna.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
    resultado = detectar_outliers(dataset[id_dataset])
    linhas = "\n".join(f"  {col}: {qtd} outlier(s)" for col, qtd in resultado.items())
    return f"Outliers detectados por coluna:\n{linhas}"

@mcp.tool()
def tool_exportar_csv(id_dataset: str, caminho_arquivo: str) -> str:
    """
    Exporta o dataset processado para um arquivo CSV no caminho especificado.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
    try:
        exportar_csv(dataset[id_dataset], caminho_arquivo)
        return f"Dataset '{id_dataset}' exportado com sucesso para '{caminho_arquivo}'."
    except Exception as e:
        return f"Erro ao exportar: {str(e)}"

@mcp.tool()
def tool_codificar_categoricas(id_dataset: str, colunas: list) -> str:
    """
    Aplica one-hot encoding nas colunas categóricas especificadas.
    """
    if id_dataset not in dataset:
        return "Erro: Dataset não encontrado."
    dataset[id_dataset] = codificar_categoricas(dataset[id_dataset], colunas)
    return f"Colunas codificadas. Colunas atuais: {list(dataset[id_dataset].columns)}"

if __name__ == "__main__":
    mcp.run()