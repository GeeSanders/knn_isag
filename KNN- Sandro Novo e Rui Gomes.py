from random import seed
from random import randrange
from csv import reader



###### ###### ###### TRATAMENTO DATASET

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

###### ###### ###### DISTÂNCIAS
##### Distância euclidiana 

def euclidiana (linha1: list ,linha2: list):
    distancia = 0.0
    for e in range(len(linha1)-1):
        distancia += (linha1[e] - linha2[e])**2
    return (distancia)**0.5

##### Distância de manhattan - Valores absolutos

def manhattan (linha1: list,linha2: list):
    distancia = 0.0  
    for m in range(len(linha1)-1):
        distancia += abs(linha1[m]-linha2[m])
    return (distancia)      

##### Distância de hamming - Calcula o XOR dos números

def hamming (linha1: list,linha2: list):
    distancia = 0.0
    assert len(linha1) == len(linha2)
    for hd1,hd2 in zip(linha1,linha2):
        if hd1 != hd2:
            distancia += hd1 + hd2
        return(distancia)

##### Função agregadora das distâncias

def get_distance_by_type(tipo: str, treino:list, calculo_row:list) -> float:
    
    distances = list()

    for row in treino:
        if tipo == 'euclidiana':
            dist = euclidiana(calculo_row , row)
        elif tipo == 'manhattan':
            dist = manhattan(calculo_row , row)
        elif tipo == 'hamming':
            dist = hamming(calculo_row , row)
        else:
            return None
        distances.append((row,dist))
        
    return distances

###### ###### ###### NORMALIZAÇÃO

##### Valors minimos e máximos - passo intermédio da normalização dos dados

def get_min_max(dataset: list):
    min_max_values = list()
    cols_list = list()
    
    # Número de colunas na linha 0
    col_length = len(dataset[0])

    # Criar lista vazia por coluna para guardar valores posteriormente
    for num in range(col_length):
        cols_list.append([])

    # Iterar linhas do dataset
    for row in dataset:
        # Iterar colunas da linha
        for col_index, col in enumerate(row):
            # Guardar valores por respectiva coluna
            cols_list[col_index].append(row[col_index])

    # Iterar nr de colunas
    for row in cols_list:
        # Min e max por coluna
        min_max_values.append((min(row), max(row)))

    return min_max_values

#### Normalização dos dados

def dataset_normalize(dataset: list, minmax: list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset

###### ###### ###### MODELO

##### Função criação dos neighbors

def gerar_neighbors(tipo: str, treino: list, teste_row:list, k: int): 
    
    distances = get_distance_by_type(tipo,treino,teste_row)
        
    distances.sort(key = lambda tup: tup[1])
    v = list()
    for i in range(k):
        v.append(distances[i][0])
    return v


###### Função previsão de classificação através do método KNN

def predict_classification(tipo:str, treino:list,teste_row:list ,k : int, normalize = False):
    
    output_values = []
    
    if normalize == True:
        min_max_treino = get_min_max(dataset = treino)
        treino_norm = dataset_normalize(treino, min_max_treino)
        for i in range(len(teste_row)):
            teste_row[i] = (teste_row[i] - min_max_treino[i][0]) / (min_max_treino[i][1] - min_max_treino[i][0]) 
        neighbors = gerar_neighbors(tipo, treino_norm, teste_row, k)
    else: 
        neighbors = gerar_neighbors(tipo, treino, teste_row, k)
    
    for row in neighbors:
        
        output_values.append(row[-1])
        
    previsao = max(set(output_values), key = output_values.count)
    
    return previsao


# Teste prático 


filename = 'iris.csv'
dataset = load_csv(filename)


for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)


# Converter categorias em números

str_column_to_int(dataset, len(dataset[0])-1)


# definir parametros do modelo

num_neighbors = 3

# definir uma linha

row = [5.4,3.4,1.5,0.4]
linha = row.copy()

# normalizar o teste  

m_m = get_min_max(dataset)
print(m_m)
normalize = True


# Efetuar previsão

label = predict_classification('manhattan',dataset, row, num_neighbors,normalize)

print('Data=%s, Predicted: %s' % (linha, label))

### Normalização da classificação :

### 2 = Iris - setosa // Normal = 1 
### 0 = Iris - versicolor // Normal = 0
### 1 = Iris - virginica // Normal = 0.5







