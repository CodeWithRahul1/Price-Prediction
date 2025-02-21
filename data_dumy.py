import pandas as pd

data = pd.read_csv("/home/admin1/Downloads/dumy_data.csv")

"""
listing_price
listing_type
listing_mileage
vehicle_year
vehicle_make
vehicle_model
vehicle_style
vehicle_exterior_color_simple
vehicle_color_interior
vehicle_interior_color_simple
vehicle_transmission_type
vehicle_transmission_speed
vehicle_drivetrain
vehicle_fuel_type
vehicle_fuel_efficiency
vehicle_specs_year
vehicle_specs_make
vehicle_specs_model
vehicle_specs_type
vehicle_specs_doors
vehicle_specs_fuel_type
vehicle_specs_engine
vehicle_specs_engine_size
vehicle_specs_transmission
vehicle_specs_transmission_type
vehicle_specs_transmission_speeds
vehicle_specs_drivetrain







1. Vehicle Information

vehicle_year: Year of the car.
vehicle_make: Car brand.
vehicle_model: Model of the car.
vehicle_trim: Trim level of the car.
vehicle_style: Style of the car (e.g., sedan, SUV).
vehicle_color_exterior: Exterior color of the car.
vehicle_exterior_color_simple: Simplified exterior color.
vehicle_color_interior: Interior color of the car.
vehicle_interior_color_simple: Simplified interior color.
vehicle_engine: Type of engine (e.g., V6, I4).
vehicle_engine_size: Size of the engine (e.g., 2.0L, 3.5L).
vehicle_engine_cylinders: Number of cylinders in the engine.
vehicle_transmission: Type of transmission (e.g., automatic, manual).
vehicle_transmission_type: Specific transmission type (e.g., CVT, DCT).
vehicle_transmission_speed: Number of transmission speeds.
vehicle_drivetrain: Drivetrain type (e.g., AWD, FWD, RWD).
vehicle_doors: Number of doors in the car.
vehicle_fuel_type: Type of fuel used (e.g., gasoline, diesel, electric).
vehicle_fuel_efficiency: General fuel efficiency.
vehicle_fu￼el_efficiency_highway: Fuel efficiency on the highway.
vehicle_fuel_efficiency_city: Fuel efficiency in the city.


2. Listing Information
listing_stock: Number of stock cars listed.
listing_price: Price of the car in the listing (target variable for prediction).
listing_mileage: Mileage of the car in the listing.
listing_type: Type of listing (e.g., for sale, auction).
listing_description: Description of the car in the listing.
listing_features: Features listed for the car.
listing_portal_urls: URLs to the listing page.

3. Seller Information
va_seller_id: Unique seller ID.
va_seller_name: Name of the seller.
va_seller_address: Sellevehicle_enginer's address.
va_seller_city: Seller's city.
va_seller_state: Seller's state.
va_seller_zip: Seller's ZIP code.
va_seller_country: Seller's country.
va_seller_latitude: Latitude of the seller.
va_seller_longitude: Longitude of the seller.

4. Other Relevant Columns
vehicle_category: General category of the car (e.g., SUV, sedan).
vehicle_title: Title of the vehicle in the listing.
vehicle_subtitle: Subtitle or additional information about the vehicle.
vehicle_specs_*: These columns provide detailed specifications, which could also help in predicting the price.
You would likely want to use columns like vehicle_year, vehicle_make, vehicle_model, vehicle_trim, listing_price (as target variable), listing_mileage, vehicle_color_exterior, vehicle_engine_size, vehicle_transmission, and vehicle_fuel_efficiency for creating a predictive model."""

listing_price
listing_mileage
listing_type
vehicle_year
days_total
vehicle_make
vehicle_style
vehicle_color_exterior
vehicle_interior_color_simple
vehicle_fuel_type
vehicle_transmission_type

selected_columns = [
    "listing_price",
    "listing_mileage",
    "listing_type",
    "vehicle_year",
    "days_total",
    "vehicle_make",
    "vehicle_style",
    "vehicle_exterior_color_simple",
    "vehicle_interior_color_simple",
    "vehicle_fuel_type",
    "vehicle_transmission_type",
    "vehicle_trim",
    "vehicle_engine_size",
    "vehicle_drivetrain",
]
