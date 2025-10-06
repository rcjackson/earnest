import xarray as xr

radars = ['KARX', 'KDMX', 'KDVN', 'KFSD', 'KOAX']
for date in range(2009, 2025):
    for radar in radars:
        try: 
            grids_KDMX = xr.open_mfdataset(
            f'/lcrc/group/earthscience/rjackson/Earnest/wind_product_b1/EARNEST.wind.{date}*.{radar}.b1.nc',
            concat_dim="time", combine="nested")
        except:
            continue
        print(f"Processing {radar} on {date}")
        grids_KDMX['spd'] = grids_KDMX['spd'] * grids_KDMX['correction_factor']
        spd_sector1 = grids_KDMX['spd'].isel(z=0).sel(
            x=slice(-31000., 0.), y=slice(-31000., 0))
        spd_sector2 = grids_KDMX['spd'].isel(z=0).sel(
            x=slice(0., 31000.), y=slice(-31000., 0))
        spd_sector3 = grids_KDMX['spd'].isel(z=0).sel(
            x=slice(-31000., 0.), y=slice(0, 31000.))
        spd_sector4 = grids_KDMX['spd'].isel(z=0).sel(
            x=slice(0., 31000.), y=slice(0., 31000.))
        spd_sector1_coarse = spd_sector1.coarsen(
                x=4, y=4, boundary="trim").mean()
        spd_sector2_coarse = spd_sector2.coarsen(
                x=4, y=4, boundary="trim").mean()
        spd_sector3_coarse = spd_sector3.coarsen(
                x=4, y=4, boundary="trim").mean()
        spd_sector4_coarse = spd_sector4.coarsen(
                x=4, y=4, boundary="trim").mean()
        out_dataset = {'spd1': (['time', 'y', 'x'], spd_sector1.values),
                       'spd2': (['time', 'y', 'x'], spd_sector2.values),
                       'spd3': (['time', 'y', 'x'], spd_sector3.values),
                       'spd4': (['time', 'y', 'x'], spd_sector4.values),
                       'spd1_c': (['time', 'yc', 'xc'],
                           spd_sector1_coarse.values),
                       'spd2_c': (['time', 'yc', 'xc'],
                           spd_sector2_coarse.values),
                       'spd3_c': (['time', 'yc', 'xc'],
                           spd_sector3_coarse.values),
                       'spd4_c': (['time', 'yc', 'xc'],
                           spd_sector4_coarse.values)}
        out_dataset = xr.Dataset(out_dataset)
        out_dataset.to_netcdf(
            f'/lcrc/group/earthscience/rjackson/Earnest/wind_quicklooks/post_processed_downscaling/{radar}_{date}.nc')
