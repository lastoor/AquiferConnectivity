# %%
import shapely
line1 = shapely.LineString([(0, 0), (100, 0)])
line2 = shapely.LineString([(0, 0), (50, 50), (100, 0)])
shapely.frechet_distance(line1, line2)
shapely.hausdorff_distance(line1, line2)
# %%
