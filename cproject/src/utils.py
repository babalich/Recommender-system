import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

#     top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()   #0.2 default
#     data = data[~data['item_id'].isin(top_popular)]

#     # Уберем самые НЕ популярные товары (их и так НЕ купят)
#     top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
#     data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                        groupby('department')['item_id'].nunique().\
                                        sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist() # 150 default
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]


    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] >= 1]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 17] # 50 default #12, 20 good result

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    # не рекоммендуем топ3 самых популярных товаров
    top = popularity.sort_values('n_sold', ascending=False)[2:take_n_popular + 2].item_id.tolist() 
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # уберем товары, которые давно не покупали
    actual_on_date = data.loc[data.week_no > 95 - 52, 'item_id'].unique()
    data.loc[~data['item_id'].isin(actual_on_date), 'item_id'] = 999999
    
    return data


def postfitlering(recomendation, N=5):
    uniques = []
    [uniques.append(item) for item in recomendation if item not in uniques]
   
    return uniques

def check_history(x, pred, history, backup_candidates, N=5):
    i = 0
    
    while len(set(pred[:N])&set(history)) == N:
        if i >= 50:
            pred[N-1] = backup_candidates[i - 10]
        else:
            pred = np.delete(pred, N-1)
        
        i += 1
    
    return pred


def featuring(data_transformed, data_features, user_features, item_features):
    data_features = data_features.merge(user_features, on='user_id', how='left')
    data_features = data_features.merge(item_features, on='item_id', how='left')
    # средняя корзина пользователя по сумме
    mean_basket_value  = data_features.groupby(['user_id']).sum()['sales_value'] \
                          /  data_features.groupby(['user_id'])['basket_id'].nunique()
    mean_basket_value = mean_basket_value.reset_index()
    mean_basket_value.columns=['user_id', 'mean_basket_value']   
    
     # средняя корзина пользователя по кол-ву
    mean_basket_quantity  = data_features.groupby(['user_id']).sum()['quantity'] \
                          /  data_features.groupby(['user_id'])['basket_id'].nunique()
    mean_basket_quantity = mean_basket_quantity.reset_index()
    mean_basket_quantity.columns=['user_id', 'mean_basket_quantity'] 
    
    ## количество уникальных магазинов
    shops_number = data_features.groupby(['user_id'])['store_id'].nunique()
    shops_number =  shops_number.reset_index()
    shops_number.columns=['user_id', 'shops_number']   
    
    ## количество уникальных товаров
    items_number = data_features.groupby(['user_id'])['item_id'].nunique()
    items_number =  items_number.reset_index()
    items_number.columns=['user_id', 'items_number']
    
    ## минимальное что-то 
    #coupon_match_disc_min = 
    coupon_match_disc = data_features.groupby(['user_id'])['coupon_match_disc'].min()
    coupon_match_disc =  coupon_match_disc.reset_index()
    coupon_match_disc.columns=['user_id', 'coupon_match_disc']
    
    # количество покупок в каждом департаменте
    user_department_count = data_features[['user_id', 'quantity', 'department']]\
                .groupby(['user_id', 'department'])['quantity'].count().reset_index()
    user_department_count['quantity'] = user_department_count['quantity'].apply(lambda x:np.log(x))
    user_department_count.rename(columns={'quantity': 'quantity_per_department'}, inplace=True)
    # средняя цена товара
    mean_item_price = data_features[['item_id', 'sales_value', 'quantity']].groupby('item_id').sum()
    mean_item_price['value'] = mean_item_price['sales_value'] / mean_item_price['quantity']
    
    # дисперсия количества покупок по депиртаментам.
    #print(data_features)
    var_by_department = data_features.groupby('department')['quantity'].var().reset_index()
    var_by_department.rename(columns={'quantity': 'var_per_department'}, inplace=True)
    
    # доля товаров по пользователю купленных со скидкой
    disc_part = data_features[['user_id']].drop_duplicates()
    disc_part = disc_part.merge(data_features.loc[data_features.retail_disc < 0]\
                                .groupby('user_id')['quantity'].count(), on='user_id', how='left')
    disc_part = disc_part.merge(data_features.groupby('user_id')['quantity'].count(), on='user_id', how='left')
    disc_part['disc_ratio'] = disc_part.iloc[:,1] / disc_part.iloc[:,2] 
    
    data_transformed = data_transformed.merge(item_features, on='item_id', how='left')
    data_transformed = data_transformed.merge(user_features, on='user_id', how='left')
    data_transformed = data_transformed.merge(mean_basket_value, on='user_id', how='left')
    data_transformed = data_transformed.merge(mean_basket_quantity, on='user_id', how='left')
    data_transformed = data_transformed.merge(shops_number, on='user_id', how='left')
    data_transformed = data_transformed.merge(items_number, on='user_id', how='left')
    data_transformed = data_transformed.merge(coupon_match_disc, on='user_id', how='left')
    
    data_transformed = data_transformed.merge(user_department_count, on=['user_id', 'department'], how='left')
    data_transformed = data_transformed.merge(mean_item_price, on='item_id', how='left')
    data_transformed = data_transformed.merge(var_by_department, on='department', how='left')
    data_transformed = data_transformed.merge(disc_part[['user_id', 'disc_ratio']], on='user_id', how='left')
    
    data_transformed['age_desc'].fillna(user_features['age_desc'].mode()[0], inplace=True)
    data_transformed['marital_status_code'].fillna(user_features['marital_status_code'].mode()[0], inplace=True)
    data_transformed['income_desc'].fillna(user_features['income_desc'].mode()[0], inplace=True)
    data_transformed['homeowner_desc'].fillna('Unknown', inplace=True)
    data_transformed['hh_comp_desc'].fillna('Unknown', inplace=True)
    data_transformed['household_size_desc'].fillna(user_features['household_size_desc'].mode()[0], inplace=True)
    data_transformed['kid_category_desc'].fillna(user_features['kid_category_desc'].mode()[0], inplace=True)    

    return data_transformed