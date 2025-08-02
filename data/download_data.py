from downloader import downloader


brands = ['Audi', 'Bmw']
view_ports = ['front', 'side', 'rear']

num_images = 1


for brand in brands:
    for view_port in view_ports:
        directory = f'dataset/{brand}/{view_port}'
        query = f'{brand} {view_port} car view'
        downloader(query, directory, num_images)