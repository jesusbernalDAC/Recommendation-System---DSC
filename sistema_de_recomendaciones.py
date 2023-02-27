import pandas as pd
import numpy as np
import hdfs as hdfs
from sklearn.model_selection import train_test_split
from entrenamiento import *
from modelo import *

def normalize_data(serie):
    return pd.qcut(serie, q=10, labels=False, duplicates='drop') + 1

if __name__ == "__main__": 
 

    # LECTURA DE LOS DATOS TRANSACCIONALES 
    transactional = pd.DataFrame()

    # FILTRO DE LAS COLUMNAS A UTILIZAR
    df = transactional.loc[:, ['Fecha', 'FechaFabricacion', 'Toneladas_CaidadePedidos', 
                      'ImpFacturadoExworksMN',  'IdFabricacion',
                      'ClaSubdireccion', 'NombreSubdireccion', 
                      'ClaZona', 'NombreZona',
                      'NombreClienteUnico', 'ClaClienteUnico', 'ClaveArticulo',
                      'NombreProducto', 'ClaGrupoEstadistico1', 'NombreGrupoEstadistico1',
                      'ClaGrupoEstadistico2', 'NombreGrupoEstadistico2',
                      'ClaGrupoEstadistico3', 'NombreGrupoEstadistico3',
                      'ClaGrupoEstadistico4', 'NombreGrupoEstadistico4', 'NombreUbicacion']]
    
    # LECTURA DE DATOS DE LA ORGANIZACIÓN 
    matrix = pd.DataFrame()
    df1 = df.merge(matrix.loc[:, ['Cuc Id', 'Cla Giro Master', 'Giro Master']], 
                left_on='ClaClienteUnico', 
                right_on='Cuc Id')

    # FILTRADO DE INFORMACIÓN POR SUBDIRECCIÓN, GIRO Y ZONA
    dftemp = df1[(df1.NombreGrupoEstadistico1.isin(["ALAMBRON", "MALLAS Y ALAMBRES", "PERFILES", "VARILLA"]))&
             (df1.NombreSubdireccion == 'MINORISTAS')&
             (df1.Fecha >= '2022-01-01')&
             (df1['Giro Master'] == 'Ferreterías')&
             (df1['NombreZona'].isin(['MIN - NUEVO LEON SUR', 'MIN - NUEVO LEON NORTE']))
            ]

    # PREPARACIÓN DE LOS DATOS PARA EL ENTRENAMIENTO
    train_test_set = dftemp.loc[:, ['ClaClienteUnico', 'NombreClienteUnico', 'ClaveArticulo', 'NombreProducto', 'Toneladas_CaidadePedidos']]

    train_test_set = train_test_set.groupby(['ClaClienteUnico', 'NombreClienteUnico', 
                        'ClaveArticulo', 'NombreProducto']).agg({'Toneladas_CaidadePedidos':'sum'}).reset_index()

    train_test_set = train_test_set.sort_values(['ClaClienteUnico', 'Toneladas_CaidadePedidos'],
                                   ascending=False).groupby(['ClaClienteUnico', 'NombreClienteUnico']).head(20).reset_index()

    cucsim = train_test_set.ClaClienteUnico.value_counts()[train_test_set.ClaClienteUnico.value_counts() >= 20].index
    train_test_set = train_test_set[train_test_set.ClaClienteUnico.isin(cucsim)]

    # ASIGNACIÓN DEL SCORE
    train_test_set['score qcut'] = train_test_set.groupby(['ClaClienteUnico', 
                                         'NombreClienteUnico']).Toneladas_CaidadePedidos.apply(normalize_data)

    train_test_set = train_test_set.groupby(['score qcut']).head(90)

    # SELECCIÓN DE LAS COLUMNAS DE USER, ITEM Y SCORE

    # Load data
    df = train_test_set[['ClaClienteUnico', 
                         'ClaveArticulo', 
                         'score qcut']].rename(columns={'ClaClienteUnico':'User',
                                                        'ClaveArticulo':'Item', 
                                                        'score qcut': 'Score'})

    users = df["User"].unique()
    items = df["Item"].unique()
    n_users = len(users)
    n_items = len(items)

    # MAPEO DE LAS CLAVES PARA EL MODELO "CUC : CUC_ID", "USER : USER_ID"
    user_to_int = {user: i for i, user in enumerate(users)}
    item_to_int = {item: i for i, item in enumerate(items)}
    int_to_item = {y: x for x, y in item_to_int.items()}
    int_to_user = {y: x for x, y in user_to_int.items()}

    df["User"] = df["User"].map(user_to_int)
    df["Item"] = df["Item"].map(item_to_int)

    # DIVISIÓN DE LOS DATOS EN TRAINING Y TESTING
    train, test = train_test_split(df, test_size=0.2)

    # ENTRENAMIENTO DEL MODELO
    best_hps = get_best_hyperparameters(train, n_users, n_items)
    model = create_ncf_model(best_hps, n_users, n_items)

    history = model.fit(
        x=[train["User"], train["Item"]],
        y=train["Score"],
        batch_size=32,
        epochs=100,
        validation_split=0.2
    )

    # EVALUACIÓN DEL MODELO
    y_true = train["Score"]
    y_pred = model.predict([train["User"], train["Item"]]).reshape(-1)
    y_pred[y_pred>= 10] = 10
    y_pred[y_pred <= 1] = 1
    mape = np.mean(np.abs((y_true - y_pred.round()) / y_true.round())) * 100
    print("MAPE: {:.2f}%".format(mape))

    # OBTENCIÓN DE LAS RECOMENDACIONES POR CLIENTE

    cuc = 0 #Example
    user_id = user_to_int[cuc]
    user_items = df[df['User'] == user_id]['Item'].unique()
    user_no_items = df[~df.Item.isin(user_items)]['Item'].unique()
    grupos = dftemp[['NombreGrupoEstadistico1', 'NombreGrupoEstadistico3', 'NombreProducto']].drop_duplicates()
    productos_comprados = train_test_set.loc[train_test_set.ClaClienteUnico == cuc, ['NombreClienteUnico', 'NombreProducto', 
                                                                                 'Toneladas_CaidadePedidos', 'score qcut']]
    productos_comprados = productos_comprados.merge(grupos, on='NombreProducto')
    scores = model.predict([np.full(len(user_no_items), user_id).reshape(-1, 1), user_no_items.reshape(-1, 1)])

    recommendations = pd.DataFrame({'item': user_no_items, 'score': scores.flatten()}).sort_values(by='score', ascending=False).head(20)
    recommendations['item'] = recommendations['item'].apply(lambda x: int_to_item[x])
    recommendations = recommendations.merge(train_test_set[['ClaveArticulo', 'NombreProducto']].drop_duplicates(), left_on = 'item', right_on='ClaveArticulo')
    recommendations = recommendations.merge(grupos, on='NombreProducto')

    print(recommendations)