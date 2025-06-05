class Product:
	all_products = {}

	def __init__(self, name, category, price, description):
		self.name = name
		self.category = category
		self.price = float(price)
		self.description = description
		Product.all_products[name.lower()] = self

	def view_info(self):
		while True:
			name = input("Enter the product name to view info: ").strip().lower()
			product = Product.all_products.get(name)
			if product:
				print(f"\n--- Product Info ---")
				print(f"Name: {product.name}")
				print(f"Category: {product.category}")
				print(f"Price: ${product.price:.2f}")
				print(f"Description: {product.description}\n")
				break
			else:
				print("No such product found.\n")

	def display_summary(self):
		while True:
			choice = input("Do you want to see a summary of the product? Yes/No: ").strip().lower()
			if choice == "yes":
				print(f"{self.name} - ${self.price:.2f}: {self.description}")
				break
			elif choice == "no":
				print("Okay.")
				break
			else:
				print("Please enter Yes or No.")

	@classmethod
	def add_product(cls):
		first_time = True
		while True:
			question = input("Add a new product? Yes/No: " if first_time else "Add another product? Yes/No: ").strip().lower()
			first_time = False

			if question == "yes":
				name = input("Enter product name: ").strip()
				if name.lower() in cls.all_products:
					print(f"A product named '{name}' already exists.\n")
					continue
				category = input("Enter product category: ").strip()
				try:
					price = float(input("Enter product price: ").strip())
				except ValueError:
					print("Invalid price. Must be a number.\n")
					continue
				description = input("Enter product description: ").strip()
				if name and category and description:
					cls(name, category, price, description)
					print(f"Product '{name}' added successfully!\n")
				else:
					print("All fields are required. Product not added.\n")
			elif question == "no":
				break
			else:
				print("Please answer with 'Yes' or 'No'.")
