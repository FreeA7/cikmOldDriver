import pandas as pd
import numpy as np

df = pd.read_csv(
    './data/training/data_train.csv', header=None)
# df[4].apply(pd.value_counts).plot(kind='bar', subplots=True)
list_level_1 = df[3].unique().tolist()
list_level_2 = df[4].unique().tolist()
list_level_3 = df[5].unique().tolist()

# ['Muslim Wear',
#  'Hand & Foot Care',
#  'Live Sound & Stage',
#  'Shampoos & Conditioners',
#  'Body and Skin Care',
#  'Traditional Laptops',
#  'Lighting & Studio Equipment',
#  'Tableware',
#  'Cookware',
#  'Women',
#  'Activity Trackers',
#  'Eyes',
#  'Shoes',
#  'Men',
#  'Phone Cases',
#  'Car Accessories',
#  'Women - Eau De Parfume',
#  'Fans',
#  'Wall Art',
#  'Moisturizers and Cream',
#  nan,
#  'Bags',
#  'Garment Steamers',
#  'Bedding Sets',
#  'Screen Protectors',
#  'Car Cameras',
#  'Cables',
#  'Media Players',
#  'Clothing',
#  'Wallets & Accessories',
#  'Makeup Accessories',
#  'Gadgets',
#  'Ink',
#  'Batteries',
#  'Gaming Headphones',
#  'Battery Adaptors',
#  'Flash Drives',
#  'Skin Care Tools',
#  'Hair Styling Appliances',
#  'Bakeware',
#  'Paper Products',
#  'Sets',
#  'Smart Watches',
#  'Projectors',
#  'Novelty',
#  'Lingerie, Sleep & Lounge',
#  'Security Cameras',
#  'Electrical',
#  'Writing Utensils',
#  'Home Entertainment',
#  'Accessories',
#  'RAM',
#  'Pillows & Bolsters',
#  'Well Being',
#  'Sports & Action Camera Accessories',
#  'Dermacare',
#  'Face',
#  'Smartwatches Accessories',
#  'Internal Hard Drives',
#  'Mac Accessories',
#  'Headphones & Headsets',
#  'Other Home Decorations',
#  'Lighting Bulbs & Components',
#  'Rugs & Carpets',
#  'Decorative Ceiling Lights',
#  'Kitchen Tools & Accessories',
#  'All-purpose',
#  'Wall Mounts & Protectors',
#  'Lips',
#  'Face Cleanser',
#  'Power Tools',
#  'Food Preparation',
#  'Bakeware & Pastries',
#  'Memory Cards',
#  'Outdoor Furniture',
#  'Mice',
#  'Kids',
#  'Hair Styling',
#  'Health Accessories',
#  'Hardware',
#  'Portable Players',
#  'PS4',
#  'Printers',
#  'Hair Treatments',
#  'Comforters, Blankets & Throws',
#  'Women - Fragrance Sets and Minis',
#  'Hair Removal Appliances',
#  'Keyboards',
#  'Sports & Action Camera',
#  'Makeup Kits, Sets, Palettes',
#  'Curtains, Blinds & Shades',
#  'Power Banks',
#  'Hair Care Accessories',
#  'Vacuum Cleaners',
#  'Gaming',
#  'Coffee Machines & Accessories',
#  'Graphic Cards',
#  'Body Lotion & Butter',
#  'Electric Kettles & Thermo Pots',
#  'Freezers',
#  'Lawn & Garden',
#  'Water Heaters',
#  'OTG Drives',
#  'Portable Speakers',
#  'Sewing Machines',
#  'Bathroom Accessories',
#  'Decorative Lamps',
#  'Clocks',
#  '2-in-1s',
#  'Shaving',
#  'Chargers',
#  'Camera Cases, Covers and Bags',
#  'Body Slimming & Electric Massagers',
#  'Safety & Security',
#  'Sports Nutrition',
#  'Specialty Lights',
#  'Power Tools Accessories',
#  'Hand Tools',
#  'Bedroom Furniture',
#  'Clothes Organizers',
#  'Kitchen & Table Linen Accessories',
#  'Monitors',
#  'Spy Cameras',
#  'Water Purifiers & Filters',
#  'Living Room Furniture',
#  'Microwaves & Ovens',
#  'Lens Accessories',
#  'Housekeeping & Laundry',
#  'Towels, Mats & Robes',
#  'Outdoor Décor',
#  'Tripods & Monopods',
#  'Adapters & Cables',
#  'Home Office Furniture',
#  'Scale & Body Fat Analyzers',
#  'Speciality Cookware',
#  'Irons',
#  'Solid State Drives',
#  'Refrigerators',
#  'Body Massage Oil',
#  'Rice Cookers & Steamers',
#  'Men - Eau De Toilette',
#  'School & Office Accessories',
#  'Electronic Cigarettes',
#  'Kitchen Storage',
#  'Foldable Wardrobes',
#  'Hoods',
#  'Toasters & Sandwich Makers',
#  'Women - Eau De Toilette',
#  'Paint Supplies',
#  'Fryers',
#  'Air Purifiers, Dehumidifiers & Humidifiers',
#  'Injury Support & Braces',
#  'Medical Tests',
#  'Outdoor Lighting',
#  'Cooking Knives',
#  'External hard Drives',
#  'Washers & Dryers',
#  'Hallway & Entry Furniture',
#  'Cooktops',
#  'Air Conditioners',
#  'Speakers',
#  'Oral Care',
#  'Routers',
#  'Nails',
#  'Bathroom Fixtures',
#  'Virtual Reality',
#  'Body Soaps & Shower Gels',
#  'LED Televisions',
#  'Beauty Supplements',
#  'Measuring & Levelling',
#  'Gaming Accessories',
#  'Kitchen & Dining Furniture',
#  'Health Monitors & Tests',
#  'Smart Televisions',
#  'Mirrorless Lenses',
#  'Walkie-Talkies',
#  'Arts & Crafts',
#  'All-In-One',
#  'Wireless USB Adapters',
#  'Coffee, Tea & Espresso',
#  'Weight Management',
#  'Point & Shoot',
#  'Macbooks',
#  'Mattresses & Protectors',
#  'Juicers & Fruit Extractors']


df[5][df[5] == "Women"] = df[4][df[5] == "Women"]
df[5][df[5] == "Gaming"] = df[4][df[5] == "Gaming"]
df[5][df[5] == "2-in-1s"] = df[4][df[5] == "2-in-1s"]
df[5][df[5] == "All-In-One"] = df[4][df[5] == "All-In-One"]



