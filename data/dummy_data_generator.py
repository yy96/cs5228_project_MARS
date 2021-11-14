import math
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


def reciprocal_fn(x, k=1, c=0):
    return k / x + c

if __name__ == '__main__':
    N = 10000
    # Warning: Remember to change user_ids zero padding if changing N
    user_ids = [f"U{i:05d}" for i in range(1, N + 1)]

    file = sys.argv[1]
    print(f"Reading data from: {file}...\n")
    data = pd.read_csv(file)

    ### Price range ###
    # Exponential distribution (salary distribution)
    # Sample mean price from exponential distribution
    # mean = k * theta
    k = 7 # shape
    theta = 11000 # scale
    target_price = np.random.gamma(k, theta, N)
    price_interval = 10000
    max_price = target_price + price_interval
    min_price = np.maximum((target_price - price_interval), 0)

    ### Filters ###
    # Filter out old cars
    mu_manufactured = -reciprocal_fn(target_price, k=4e5, c=-2015)
    sigma_manufactured = mu_manufactured / 1000
    limit_manufactured = np.abs(np.random.normal(
        mu_manufactured, sigma_manufactured, N
    )).astype(int)

    # Filter out cars with high mileage
    mu_mileage = reciprocal_fn(target_price, 5e9, 8e4)
    sigma_mileage = mu_mileage / 15
    limit_mileage = np.abs(np.random.normal(
        mu_mileage, sigma_mileage, N
    ))

    # Filter based on no. of owners
    pi = np.array([.2, .3, .2, .15, .1, .05])
    limit_no_of_owners = np.random.choice(np.arange(1, 7), N, p=pi)

    # TODO: Filter out CAT B cars (whether to filter based on price)

    ## No. of viewings
    k_views = 2.5
    theta_views = 10
    views = np.random.gamma(k_views, theta_views, N).astype(int)

    ## Preference for make
    mu_preference = .7
    sigma_preference = .05
    make_preference = np.minimum(1, np.abs(np.random.normal(
        mu_preference, sigma_preference, N
    )))
    others_preference = 1 - make_preference
    n_views_make = (views * make_preference).astype(int)
    n_views_others = (views * others_preference).astype(int)

    parameters = pd.DataFrame({
        'target_price': target_price,
        'max_price': max_price,
        'min_price': min_price,
        'mu_manufactured': mu_manufactured,
        'limit_manufactured': limit_manufactured,
        'mu_mileage': mu_mileage,
        'limit_mileage': limit_mileage,
        'limit_no_of_owners': limit_no_of_owners,
        'views': views,
        'n_views_make': n_views_make,
        'n_views_others': n_views_others
    }, index=user_ids)

    mu_duration = 40
    sigma_duration = 10
    list_userdata = []
    for row in parameters.itertuples():
        data_price = data[np.logical_and(
            data.price < row.max_price,
            data.price > row.min_price
        )]
        data_fltr = data_price[data_price.manufactured >= row.limit_manufactured]
        data_fltr = data_fltr[data_fltr.mileage < row.limit_mileage]
        data_fltr = data_fltr[data_fltr.no_of_owners <= row.limit_no_of_owners]
        
        preferred_make = np.random.choice(data_fltr.make, 1).item()
        data_preferred = data_fltr[data_fltr.make == preferred_make]
        data_others = data_fltr[data_fltr.make != preferred_make]
        
        if data_preferred.shape[0] < row.n_views_make:
            print((
                f'{data_preferred.shape[0]} listings of {preferred_make} '
                f'but {row.n_views_make} needed!'
            ))
            print("[INFO] Simulating user with no preference for any make instead.")
            
            if data_fltr.shape[0] < row.views:
                print(f'{data_fltr.shape[0]} listings but {row.views} needed!')
                print("[INFO] Removing all filters except for price range.")
                listings = np.random.choice(
                    data_price.listing_id, row.views, replace=False
                )
            else:
                listings = np.random.choice(
                    data_fltr.listing_id, row.views, replace=False
                )
                
        else:
        	## Users with preference for a single make
            listings_preferred = np.random.choice(
                data_preferred.listing_id, row.n_views_make, replace=False
            )

            if data_others.shape[0] < row.n_views_others:
                raise ValueError(
                    "Not enough listings from other makes after filtering!"
                )

            listings_others = np.random.choice(
                data_others.listing_id, row.n_views_others, replace=False
            )
            listings = np.concatenate([listings_preferred, listings_others])
        
        durations = np.abs(np.random.normal(mu_duration, sigma_duration, listings.size))
        user_ids = np.repeat(row.Index, listings.size)
        user_data = pd.DataFrame({
            'user_id': user_ids,
            'listing_id': listings,
            'duration': durations
        })
        list_userdata.append(user_data)


    viewing_data = pd.concat(list_userdata, ignore_index=True)
    viewing_data.to_csv(
        "viewing_data.tsv",
        index=False,
        float_format="%.1f"
    )
    print("\nFinished generating dummy data!")