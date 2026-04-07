import os
os.environ["MPLBACKEND"] = "Agg"

import osmnx as ox
import folium
import pandas as pd
import numpy as np
import branca.colormap as cm

# Definition of Area

place = "Oxford city centre, Oxford, UK"

# Loading Street Graph

G = ox.graph_from_place(place, network_type="all")
edges = ox.graph_to_gdfs(G, nodes=False)

edges_proj = edges.to_crs(epsg=3857)
center_proj = edges_proj.geometry.centroid
center_wgs = center_proj.to_crs(epsg=4326)
center = [center_wgs.y.mean(), center_wgs.x.mean()]

# Loading Buildings

buildings = ox.features_from_place(place, tags={"building": True})
buildings = buildings[buildings.geometry.type == "Polygon"]
buildings = buildings.to_crs(epsg=3857)
buildings["area_m2"] = buildings.geometry.area

# Classification of Building Types

def classify_building(row):
    b = str(row.get("building")).lower()
    amenity = str(row.get("amenity")).lower()
    shop = str(row.get("shop")).lower()
    office = str(row.get("office")).lower()
    tourism = str(row.get("tourism")).lower()

    if b in ["house", "residential", "apartments", "detached", "terrace"]:
        return "residential"
    elif b in ["commercial", "retail", "office", "warehouse"] or shop != "nan" or office != "nan":
        return "commercial"
    elif tourism in ["hotel"] or amenity in ["restaurant", "cafe", "bank", "fast_food"]:
        return "commercial"
    elif b in ["school", "university", "college"] or amenity in ["school", "university", "college", "library", "kindergarten"]:
        return "education"
    else:
        return "other"

buildings["type"] = buildings.apply(classify_building, axis=1)

# Building Type Distribution

counts = buildings["type"].value_counts()
percent = buildings["type"].value_counts(normalize=True) * 100

df_counts = pd.DataFrame({
    "count": counts,
    "percent": percent.round(1)
})

print("\nGebäudetypen-Verteilung:")
print(df_counts)

# Levels (OSM + Fallback)

buildings["levels"] = pd.to_numeric(buildings.get("building:levels"), errors="coerce")

avg_levels = {
    "residential": 2.75,
    "commercial": 3,
    "education": 3,
    "other": 3.5
}

def fill_levels(row):
    return row["levels"] if pd.notnull(row["levels"]) else avg_levels[row["type"]]

buildings["levels"] = buildings.apply(fill_levels, axis=1).clip(1, 10)
buildings["total_area_m2"] = buildings["area_m2"] * buildings["levels"]

min_total_area = 50  # m² 
buildings = buildings[buildings["total_area_m2"] >= min_total_area]


# Calculation of Energy Use

EUI = {
    "residential": 180,
    "commercial": 250,
    "education": 220,
    "other": 150
}

heating_share = {
    "residential": 0.65, # guess
    "commercial": 0.45, # guess
    "education": 0.60, # guess
    "other": 0.50, # guess
}

buildings["heating_energy"] = buildings.apply(
    lambda x: x["total_area_m2"] * EUI[x["type"]] * heating_share[x["type"]] / 365,
    axis=1
)

# Heating energy per building

energy_shares = {
    "other":      {"Gas": 0.8, "Electric": 0.1, "Renewable": 0.05, "Divers": 0.05},
    "education":  {"Gas": 0.85, "Electric": 0.10, "Renewable": 0.03, "Divers": 0.02},
    "residential":{"Gas": 0.75, "Electric": 0.12, "Renewable": 0.07, "Divers": 0.06},
    "commercial": {"Gas": 0.90, "Electric": 0.06, "Renewable": 0.02, "Divers": 0.02},
}

def assign_heating_type(row):
    shares = energy_shares[row["type"]]
    return np.random.choice(list(shares.keys()), p=list(shares.values()))

buildings["heating_type"] = buildings.apply(assign_heating_type, axis=1)

# CO2 before

co2e_factors = {
    "Gas": 0.203, # conversion factor for UK, Natural Gas
    "Electric": 0.177, # conversion factor for UK 
    "Renewable": 0.02, # guess
    "Divers": 0.185 # guess
}

def calc_co2(row):
    return (row["heating_energy"] / 365) * co2e_factors[row["heating_type"]]

buildings["CO2_before"] = buildings.apply(calc_co2, axis=1)

# 1ENERGY Heating Network

buildings["netz_connected"] = False

buildings.loc[buildings["type"].isin(["education", "commercial"]), "netz_connected"] = True
res_subset = buildings[buildings["type"] == "residential"].sample(frac=0.5, random_state=42).index
buildings.loc[res_subset, "netz_connected"] = True

co2_factor_network = 0.08 # guess

def calc_co2_after(row):
    current_factor = co2e_factors[row["heating_type"]]
    
    # Only connect, if the netz-connected factor is better than current
    if row["netz_connected"] and co2_factor_network < current_factor:
        return (row["heating_energy"] / 365) * co2_factor_network
    else:
        return (row["heating_energy"] / 365) * current_factor

buildings["CO2_after"] = buildings.apply(calc_co2_after, axis=1)

# Table

energy_summary = []

for btype in ["residential", "commercial", "education", "other"]:
    subset = buildings[buildings["type"] == btype]
    
    energy_summary.append({
        "Type": btype,
        "Gas_kWh_per_day": subset[subset["heating_type"]=="Gas"]["heating_energy"].sum(),
        "Electric_kWh_per_day": subset[subset["heating_type"]=="Electric"]["heating_energy"].sum(),
        "Renewable_kWh_per_day": subset[subset["heating_type"]=="Renewable"]["heating_energy"].sum(),
        "Divers_kWh_per_day": subset[subset["heating_type"]=="Divers"]["heating_energy"].sum(),
        "Total_kWh_per_day": subset["heating_energy"].sum()
    })

energy_df = pd.DataFrame(energy_summary)

print("\nHEIZENERGIE nach Gebäudetyp und Heizsystem:")
print(energy_df)

# Aggregation

co2_by_type_before = buildings.groupby("type")["CO2_before"].sum()
co2_by_type_after = buildings.groupby("type")["CO2_after"].sum()


# Maps

buildings_4326 = buildings.to_crs(epsg=4326)

# Before Map

m_before = folium.Map(location=center, zoom_start=15)
colormap_before = cm.linear.YlOrRd_09.scale(0, buildings_4326["CO2_before"].max())

for _, row in buildings_4326.iterrows():
    popup = f"""
    <b>Type:</b> {row['type']}<br>
    <b>Area (m²):</b> {row['area_m2']:.1f}<br>
    <b>Levels:</b> {row['levels']}<br>
    <b>Total Area (m²):</b> {row['total_area_m2']:.1f}<br>
    <b>Heating system:</b> {row['heating_type']}<br>
    <b>Heating energy (kWh/day):</b> {row['heating_energy']/365:.1f}<br>
    <b>CO2 per building (kg/day):</b> {row['CO2_before']:.1f}<br>
    <b>Total CO2 for {row['type']} (kg/day):</b> {co2_by_type_before[row['type']]:.1f}
    """
    
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda feature, co2=row["CO2_before"]: {
            'fillColor': colormap_before(co2),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7
        },
        tooltip=popup
    ).add_to(m_before)

colormap_before.add_to(m_before)
m_before.save("oxford_CO2_before.html")

# After Map

m_after = folium.Map(location=center, zoom_start=15)
colormap_after = cm.linear.YlOrRd_09.scale(0, buildings_4326["CO2_after"].max())

for _, row in buildings_4326.iterrows():
    popup = f"""
    <b>Type:</b> {row['type']}<br>
    <b>Area (m²):</b> {row['area_m2']:.1f}<br>
    <b>Levels:</b> {row['levels']}<br>
    <b>Total Area (m²):</b> {row['total_area_m2']:.1f}<br>
    <b>Heating system:</b> {row['heating_type']}<br>
    <b>Connected to network:</b> {row['netz_connected']}<br>
    <b>Heating energy (kWh/day):</b> {row['heating_energy']/365:.1f}<br>
    <b>CO2 per building (kg/day):</b> {row['CO2_after']:.1f}<br>
    <b>Total CO2 for {row['type']} (kg/day):</b> {co2_by_type_after[row['type']]:.1f}
    """
    
    folium.GeoJson(
        row.geometry.__geo_interface__,
        style_function=lambda feature, co2=row["CO2_after"]: {
            'fillColor': colormap_after(co2),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7
        },
        tooltip=popup
    ).add_to(m_after)

colormap_after.add_to(m_after)
m_after.save("oxford_CO2_after.html")

print("Maps saved!")

#  Comparison

total_energy_day = buildings["heating_energy"].sum() / 365

total_co2_before = buildings["CO2_before"].sum()
total_co2_after = buildings["CO2_after"].sum()

print("\n==============================")
print(" GESAMTVERGLEICH")
print("==============================")
print(f"Total heating energy (kWh/day): {total_energy_day:,.0f}")
print(f"CO2 BEFORE (kg/day): {total_co2_before:,.0f}")
print(f"CO2 AFTER (kg/day): {total_co2_after:,.0f}")
print(f"CO2 SAVINGS (kg/day): {total_co2_before - total_co2_after:,.0f}")
print(f"CO2 REDUCTION (%): {100*(total_co2_before - total_co2_after)/total_co2_before:.1f}%")

# Comparison by type

summary = []

for btype in ["residential", "commercial", "education", "other"]:
    
    subset = buildings[buildings["type"] == btype]
    
    energy_day = subset["heating_energy"].sum() / 365
    co2_before = subset["CO2_before"].sum()
    co2_after = subset["CO2_after"].sum()
    
    summary.append({
        "Type": btype,
        "Energy_kWh_per_day": energy_day,
        "CO2_before_kg_per_day": co2_before,
        "CO2_after_kg_per_day": co2_after,
        "CO2_savings_kg_per_day": co2_before - co2_after,
        "Reduction_%": 100*(co2_before - co2_after)/co2_before if co2_before > 0 else 0
    })

summary_df = pd.DataFrame(summary)

# Example residential

house_area_m2 = 100       # guess for a typical house
levels = 2                 
total_area = house_area_m2 * levels

EUI_residential = 180       
heating_share = 0.65       
COP_heatpump = 4.3

# Before in case of gas heating

heating_energy_yr = total_area * EUI_residential * heating_share  
co2_factor_gas = 0.203
cost_gas_per_kWh = 0.10 # guess

co2_before = heating_energy_yr * co2_factor_gas / 365  # kg/day
cost_before_day = heating_energy_yr * cost_gas_per_kWh / 365  

# With district heating
co2_factor_network = 0.04
network_fee = 0.03
heat_price_customer = 0.13

electricity_needed_yr = heating_energy_yr / COP_heatpump
cost_after_day = (electricity_needed_yr * heat_price_customer / 365) + (heating_energy_yr * network_fee / 365)
co2_after = heating_energy_yr * co2_factor_network / 365

# Monthly costs and savings
cost_before_month = cost_before_day * 30
cost_after_month = cost_after_day * 30
savings_month = cost_before_month - cost_after_month
reduction_pct = 100 * (co2_before - co2_after) / co2_before

print("\n==============================")
print("Single House Example")
print("==============================")
print(f"Area: {total_area} m² ({levels} Levels)")
print(f"Heating Energy: {heating_energy_yr:,.0f} kWh")
print(f"CO2 before: {co2_before*365:.1f} kg/Year")
print(f"CO2 after: {co2_after*365:.1f} kg/Year")
print(f"CO2 Savings: {reduction_pct:.0f}%")
print(f"Costs before (Gas): £{cost_before_month:.2f}/Month")
print(f"Costs after (Network + HP): £{cost_after_month:.2f}/Month")
print(f"Monthly Savings: £{savings_month:.2f}")
print(f"Percentage Savings: {100 * savings_month / cost_before_month:.1f}%")








