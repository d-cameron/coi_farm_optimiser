mod crop;

use crop::*;

fn main() {
    println!("Captain of Industry Food Production Optimizer");
    
    // User input: crop consumption requirements
    // The units are in number of farms required for each crop calculated from another calculator (e.g. factoriolab)
    // To prevent rounding errors, use at least two decimal places for your inputs
    // INPUT IS IN NUMBER OF FARMS NEEDED FOR EACH CROP NOT ITEMS/MIN!!!
    let number_of_farms_needed_for_each_crop: [f32; NUM_CROPS] = [
        82.31,    // Potato
        309.65,   // Corn
        181.04,   // Wheat
        68.35,    // Soybean
        112.79,   // Sugarcane
        159.13,   // Vegetables
        199.48,   // Fruit 
        59.83,    // Canola
        117.58,   // Poppy
        0.0,      // Flowers (excluded)
        0.0,      // Sapling (excluded) 
        0.0,      // Green Manure (Currently, fertility is not calculated so this should always be zero)
        0.0, 0.0, 0.0, 0.0  // Unused crops (excluded)
    ];
    // TODO: handle animal feed from non-corn. Need to:
    // 1) Input the amount of corn that is used for animal feed
    // 2) Adjust the corn requirement accordingly
    // 3) Calculate the min ratio for the 4 animal feed crops ignoring animal feed production
    // 4) Calculate the animal feed production
    // 5) Pro-rata consume all 4 feedstocks in feed-output-adjusted at-ratio consumption rates until feed ratio == pro-rata reduced foodstock ratio
    //  Note that step 5 only kicks in if the animal feed production ratio is less than the feedstock ratio calculated in step 3).
    //  That is, if we already have excess feed from the 3 crops with excess production, there's no need to make feed from the 4th.
    
    // Create configuration from user inputs
    let config = CropConfig::new(&number_of_farms_needed_for_each_crop);
    
    // Test optimization for 3, 4, and 5 farms
    for n_farms in 3..=10 {
        println!("\n{:=<80}", "");
        println!("OPTIMIZING FOR {} FARMS", n_farms);
        println!("{:=<80}", "");
        
        let solution = optimize_farm_allocation(n_farms, &config);
        
        println!("\nBest solution found:");
        println!("  Effective production: {:.4} {:.4}%", solution.effective_production, solution.effective_production / n_farms as f32);
        
        let total_production = calculate_total_production(&solution.farm_rotation_indices, &config);
        let bottleneck_crop = solution.bottleneck_crop_index(&config);
        
        println!("  Bottleneck crop: {} (production: {:.4}, ratio: {:.4})", 
                 CROP_NAMES[bottleneck_crop], 
                 total_production[bottleneck_crop], 
                 config.crop_ratios[bottleneck_crop]);
        
        println!("\nFarm allocations:");
        for (farm_id, &rotation_idx) in solution.farm_rotation_indices.iter().enumerate() {
            let rotation = &config.all_rotations[rotation_idx];
            println!("  Farm {} (rotation {}): {:?}", 
                     farm_id + 1, rotation_idx, 
                     rotation.crop_indices.iter()
                         .map(|&i| CROP_NAMES[i]).collect::<Vec<_>>());
        }
        
        println!("\nTotal production by crop:");
        for (i, &production) in total_production.iter().enumerate() {
            if config.crop_ratios[i] > 0.0 || production > 0.0 {
                let effective = if config.crop_ratios[i] > 0.0 { 
                    production / config.crop_ratios[i] 
                } else { 
                    f32::INFINITY 
                };
                println!("  {}: {:.4} (need: {:.4}, effective: {:.4})", 
                         CROP_NAMES[i], production, config.crop_ratios[i], effective);
            }
        }
    }
}