def prefilter_items(data_train, item_features, departpments_):
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(5100).item_id.tolist()
    #добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top), 'item_id'] = 999999 
    
    # Уберем самые популярные 
    data_train.loc[data_train['item_id'].isin(top[:100]), 'item_id'] = 999999 
    
    # Уберем самые непопулряные 
    # не совсем понял, уберем самые неполпулярные из популярных? Разделить вне топ-5000 от хвоста топ-5000?
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    timeline = data_train['week_no'].max() - 54
    date_filter = data_train['item_id'].loc[data_train['week_no'] >= timeline].unique()
    data_train = data_train.loc[data_train['item_id'].isin(date_filter)]
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    
    departments_filter = item_features.loc[~item_features['department'].isin(departpments_), 'item_id'].unique()
    data_train = data_train.loc[~data_train['item_id'].isin(departments_filter)]
    
     
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data_train = data_train[data_train.sales_value > 1.5]
    
    # Уберем слишком дорогие товарыs
    price_filter = data_train['sales_value'].quantile(0.90)
    data_train = data_train[data_train.sales_value < price_filter]
    # ...
    
    return data_train
def postfilter_items():
    pass

