from Product import Product

# Add a few sample products
Product("Bag", "Women items", 5, "yellow and classy")
Product("Watch", "Accessories", 25, "Leather strap, waterproof")

# Interact with the product
p = Product.all_products["bag"]
p.view_info()
p.display_summary()

# Add more products via input
Product.add_product()
