# Urban Morphology Analysis: Building Density and Orientation in Buenos Aires

## Project Overview

I created this repository to implement a geospatial processing pipeline. My code extracts and analyzes urban morphological characteristics from building polygon data, specifically calculating building density and orientation for Buenos Aires using Google's Open Buildings dataset.

## Methodology

### Data Acquisition
- City boundary: I retrieved this from the official open data portal of Buenos Aires
- Building footprints: I extracted these from Google's Open Buildings dataset via Earth Engine API

### Building Density Calculation

The implementation follows these steps to calculate building density:
1. Create a regular 10m × 10m grid over the Buenos Aires area
2. For each grid cell, calculate the building density as:
   ```
   density = total_building_area_within_cell / cell_area
   ```
3. The result is between 0.0 (no buildings) and 1.0 (completely covered)

To optimize performance, I:
- Used spatial indexing to identify candidate buildings for each cell
- Applied different processing strategies based on building size:
   - Small buildings (≤25% of cell area): Uses centroid containment for faster calculation
   - Large buildings (>25% of cell area): Performs geometric intersection

### Building Orientation Analysis

For building orientation, the implementation:
1. Calculates individual building orientation using OpenCV's minimum area rectangle function
2. Normalizes angles to the 0°-180° range
3. Aggregates orientations at the grid cell level using vector averaging:
   - Converts angles to radians and doubles them (×2)
   - Calculates mean cosine and sine components
   - Converts back to degrees and divides by 2, handling the circular nature of orientation data

I chose this approach to properly handle the issue of angle circularity (e.g., 0° and 180° representing the same orientation).

## Implementation Details

### Output Files
My script produces:
- Grid files in UTM and WGS84 formats
- Result files with density and orientation metrics (both UTM and WGS84)

### Technical Considerations
- Proper handling of CRS transformations between WGS84 and UTM
- Use of spatial indexing and batch processing

## Implementation Challenges and Technical Decisions

### Performance vs. Accuracy Trade-offs

I made several practical compromises to balance computational efficiency with analysis quality:

1. **Confidence Score Filtering (0.75 threshold)**
   - **Why implemented**: I needed to reduce the dataset size, improving processing speed
   - **Drawback**: Creates significant bias against informal settlements where building detection confidence is typically lower (this is my hypothesis)
   - **Better alternative**: Process all buildings regardless of confidence

2. **Small Building Approximation**
   - **Why implemented**: Using centroid containment for small buildings (≤25% of cell area) significantly reduces computation time
   - **Drawback**: Introduces small errors at grid cell boundaries
   - **Justification**: I needed to run the code faster

3. **Hard-coded UTM Zone**
   - **Why implemented**: Buenos Aires-specific optimization using UTM Zone 21S
   - **Limitation**: Makes the code less reproducible to other regions

5. **Simplified Orientation Calculation**
   - **Why implemented**: Uses OpenCV's minimum area rectangle for computational efficiency
   - **Limitation**: Less accurate for irregular buildings common in informal settlements
   - **Better approach**: Use principal component analysis (PCA) or more sophisticated shape analysis


# Extra: Viz documentation

This document describes a simple visualization tool I created to manually validate the building density and orientation metrics calculated in the main analysis pipeline.
I chose the area near my house to do a "manual validation" over an area I know.

## Overview

The visualization creates an interactive web map of a 200×200m sample area in Buenos Aires, displaying:
- The analysis grid cells colored by building density
- Building footprints from Google Open Buildings
- Satellite imagery for context
- Tooltips with density and orientation values

## Known Limitations

### Data Representation Issues

1. **Grid Cell Alignment**: The grid cells don't perfectly align with the sample area boundaries, causing some edge cells to be included that only partially intersect the sample area

2. **Building Coverage Issues**: Some buildings near the boundary may be excluded due to the simple intersection-based filtering

3. **Confidence Threshold Bias**: The visualization filters buildings using the same 0.75 confidence threshold as the main analysis, perpetuating the bias against informal settlements