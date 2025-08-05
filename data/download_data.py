from downloader import downloader


brands = ['Audi', 'Mercedes', 'Toyota', 'Fiat']
view_ports = ['front', 'side', 'rear']

num_images = 50


for brand in brands:
    for view_port in view_ports:
        directory = f'dataset/{brand}/{view_port}_photo'
        query = f'{brand} {view_port} car real photo'
        downloader(query, directory, num_images)