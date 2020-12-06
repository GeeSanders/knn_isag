# Trabalho-KNN-

Criação de um algoritmo KNN no âmbito da disciplina de programação da Pós Graduação em Data Science & Business Intelligence


O K-Nearest Neighbour é uma técnicas mais simples aprendizagem supervisionada de Machine Learning. 
Esta técnica é usada para problemas de classificação, sendo um método não paramétrico não faz nenhuma assunção relativamente aos dados, um dos pressupostos para resolver problemas de escala, é a normalização dos dados. 
O objetivo é encontrar a classe de uma linha de dados através deste método.

Passando ao código e ao raciocínio efetuado: 

### Cálculo das distâncias: 

O objetivo nestas funções, é ter como ponto de referência uma qualquer linha do nosso dataset, por forma a serem calculadas as distâncias a partir dessa linha. A partir do cálculo dessas distâncias conseguiremos mais à frente averiguar a classe da nossa observação. 

```python
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

```
O objetivo aqui é o argumento treino ser o dataset inteiro e o calculo_row ser o argumento referente à linha de referência das distâncias. Esta função calculará a distância de acordo com a passagem do input no argumento tipo.

###  Normalização do dataset 

A normalização do dataset é efetuada através do método min max. X = (x - xmin)(xmax-xmin), o objetivo aqui é as observações tomarem apenas valores compreendidos entre 0 e 1. 

Isto foi conseguido em duas fases, 
1º Cálculo dos valores minimos e máximos: 
```python
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
`````
2º Calcular para cada valor de cada coluna o valor normalizado
```python

def dataset_normalize(dataset: list, minmax: list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    return dataset
    
```

###  Modelo de Classificação

Após o calculo das distâncias será efetuado a ordenação das mesmas por forma a serem agrupadas as observações mais parecidas entre si, o mesmo é atingido através da seguinte função, aqui terá em conta o tipo de distância calculada, o dataset como todo (treino), a observação teste e ainda o número de vizinhos para o cálculo, o k. 

```python


def gerar_neighbors(tipo: str, treino: list, teste_row:list, k: int): 
    
    distances = get_distance_by_type(tipo,treino,teste_row)
        
    distances.sort(key = lambda tup: tup[1])
    v = list()
    for i in range(k):
        v.append(distances[i][0])
    return v

`````

Por fim a função que calcula a classificação da observação será a seguinte: 

```python

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



`````

Esta função leva cinco argumentos, o tipo refere-se à distância, o treino, refere-se ao dataset como um todo, o teste_row refere-se à observação para a qual queremos calcular a classificação, o k refere-se a quantidade de neighbors a utilizar no nosso cálculo e ainda o argumento do tipo boolean normalize, que permite ao utilizador a normalização das observações. 
Assim a previsão será igual ao valor da classificação dos vizinhos mais próximos dependendo do k dado. 


###  Aplicação prática no dataset 'iris.csv' 

O primeiro passo é a conversão da variável categórica em variável numérica: 


```python

filename = 'iris.csv'
dataset = load_csv(filename)


for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)


str_column_to_int(dataset, len(dataset[0])-1)

`````
Obtivemos os seguintes valores : 
2 = Iris - setosa
0 = Iris - versicolor
1 = Iris - virginica

Para a nossa experiência escolhemos o num_neighbors = 3 
E definimos os seguintes dados para o teste row = [5.4,3.4,1.5,0.4]
Optámos também pela normalização das observações, assim normalize = True


Efetuando-se a previsão propriamente dita: 


```python

label = predict_classification('manhattan',dataset, row, num_neighbors,normalize)

print('Data=%s, Predicted: %s' % (row, label))



`````

Obtivemos o seguinte resultado

Data=[5.4, 3.4, 1.5, 0.4], Predicted: 1.0

À esquerda temos a observação testada, à direita temos a classe da previsão, verificamos a normalização dos valores das classes pelo que embora o valor seja 1.0, os valores normalizados são descritos como segue:

Iris - setosa Normal = 1  

Iris - versicolor  Normal = 0

Iris - virginica  Normal = 0.5


Após o teste verificamos que efetivamente esta observações pertence à classe Iris-setosa. 

