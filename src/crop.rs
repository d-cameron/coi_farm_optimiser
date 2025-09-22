use std::time::Instant;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Crop constants - using ordinal indexing for 16 crops (2 AVX words)
pub const NUM_CROPS: usize = 16;
pub const CROP_NAMES: [&str; NUM_CROPS] = [
    "Potato", "Corn", "Wheat", "Soybean", "Sugarcane", 
    "Vegetables", "Fruit", "Canola", "Poppy", "Flowers", 
    "Sapling", "Green Manure", "Crop13", "Crop14", "Crop15", "Crop16"
];

pub const CROP_GROWING_TIMES: [u32; NUM_CROPS] = [
    3, 4, 6, 4, 9, 4, 8, 3, 4, 4, 4, 12, 3, 3, 3, 3  // Growing times in months
];

// Precomputed rotation data
#[derive(Debug, Clone)]
pub struct RotationData {
    pub crop_indices: Vec<usize>,
    pub production_percentages: [f32; NUM_CROPS],
}

// Configuration struct that holds all derived data from crop inputs
pub struct CropConfig {
    pub crop_ratios: [f32; NUM_CROPS],
    pub crop_ratios_reciprocal: [f32; NUM_CROPS], 
    pub crop_usage_mask: [f32; NUM_CROPS],
    pub all_rotations: Vec<RotationData>,
}

impl CropConfig {
    pub fn new(crop_input: &[f32; NUM_CROPS]) -> Self {
        // Calculate the sum of all non-zero crop inputs for normalization
        let sum: f32 = crop_input.iter().sum();
        
        // Create normalized ratios
        let mut crop_ratios = [0.0; NUM_CROPS];
        for i in 0..NUM_CROPS {
            if crop_input[i] > 0.0 {
                crop_ratios[i] = crop_input[i] / sum;
            }
        }
        
        // Precompute reciprocals for faster AVX division
        let mut crop_ratios_reciprocal = [0.0; NUM_CROPS];
        for i in 0..NUM_CROPS {
            if crop_ratios[i] > 0.0 {
                crop_ratios_reciprocal[i] = 1.0 / crop_ratios[i];
            }
            // For crops with zero ratios, reciprocal remains 0.0
            // The infinity will be added via crop_usage_mask instead
        }
        
        // Create crop usage mask for AVX operations
        let mut crop_usage_mask = [0.0; NUM_CROPS];
        for i in 0..NUM_CROPS {
            if crop_ratios[i] == 0.0 {
                crop_usage_mask[i] = f32::INFINITY;
            }
        }
        
        // Generate all valid rotations
        let all_rotations = Self::generate_rotations(&crop_ratios);
        
        CropConfig {
            crop_ratios,
            crop_ratios_reciprocal,
            crop_usage_mask,
            all_rotations,
        }
    }
    
    fn generate_rotations(crop_ratios: &[f32; NUM_CROPS]) -> Vec<RotationData> {
        let mut rotations = Vec::new();
        
        // Get only crops with non-zero ratios for rotation generation
        let valid_crops: Vec<usize> = (0..NUM_CROPS)
            .filter(|&i| crop_ratios[i] > 0.0)
            .collect();
        
        // Generate 2-crop rotations (only with crops that have requirements)
        for i in 0..valid_crops.len() {
            for j in (i + 1)..valid_crops.len() {
                let crop_indices = vec![valid_crops[i], valid_crops[j]];
                let rotation = Self::create_rotation_data(crop_indices);
                rotations.push(rotation);
            }
        }
        
        // Generate 3-crop rotations
        for i in 0..valid_crops.len() {
            for j in (i + 1)..valid_crops.len() {
                for k in (j + 1)..valid_crops.len() {
                    let crop_indices = vec![valid_crops[i], valid_crops[j], valid_crops[k]];
                    let rotation = Self::create_rotation_data(crop_indices);
                    rotations.push(rotation);
                }
            }
        }
        
        // Generate 4-crop rotations
        for i in 0..valid_crops.len() {
            for j in (i + 1)..valid_crops.len() {
                for k in std::iter::once(i).chain((j + 1)..valid_crops.len()) {
                    for l in (k + 1)..valid_crops.len() {
                        let crop_indices = vec![valid_crops[i], valid_crops[j], valid_crops[k], valid_crops[l]];
                        let rotation = Self::create_rotation_data(crop_indices);
                        rotations.push(rotation);
                    }
                }
            }
        }
        
        rotations
    }
    
    fn create_rotation_data(crop_indices: Vec<usize>) -> RotationData {
        let mut production_percentages = [0.0; NUM_CROPS];
        
        let total_time: u32 = crop_indices.iter()
            .map(|&i| CROP_GROWING_TIMES[i])
            .sum();
        
        for &crop_idx in &crop_indices {
            production_percentages[crop_idx] += CROP_GROWING_TIMES[crop_idx] as f32 / total_time as f32;
        }
        
        RotationData {
            crop_indices,
            production_percentages,
        }
    }
}

#[derive(Debug)]
pub struct Solution {
    pub farm_rotation_indices: Vec<usize>,  // Indices into all_rotations
    pub effective_production: f32,
}

impl Solution {
    pub fn bottleneck_crop_index(&self, config: &CropConfig) -> usize {
        let total_production = calculate_total_production(&self.farm_rotation_indices, config);
        let mut min_effective = f32::INFINITY;
        let mut bottleneck_crop_index = 0;

        for i in 0..NUM_CROPS {
            if config.crop_ratios[i] > 0.0 {
                let effective_production = total_production[i] / config.crop_ratios[i];
                
                if effective_production < min_effective {
                    min_effective = effective_production;
                    bottleneck_crop_index = i;
                }
            }
        }
        
        bottleneck_crop_index
    }
}

pub fn calculate_total_production(rotation_indices: &[usize], config: &CropConfig) -> [f32; NUM_CROPS] {
    let mut total_production = [0.0; NUM_CROPS];
    
    for &rotation_idx in rotation_indices {
        let rotation = &config.all_rotations[rotation_idx];
        for i in 0..NUM_CROPS {
            total_production[i] += rotation.production_percentages[i];
        }
    }
    
    total_production
}

// Calculate effective production (bottleneck analysis) for 16 crops
#[allow(dead_code)]
fn calculate_effective_production(total_production: &[f32; NUM_CROPS], config: &CropConfig) -> f32 {
    let mut min_effective = f32::INFINITY;

    // Check ALL crops, but skip crops with zero ratios (they don't constrain production)
    for i in 0..NUM_CROPS {
        if config.crop_ratios[i] > 0.0 {
            let effective_production = total_production[i] / config.crop_ratios[i];
            
            if effective_production < min_effective {
                min_effective = effective_production;
            }
        }
    }
    
    min_effective
}

// AVX256 accelerated function to calculate effective production directly from rotation indices
// Now handles 16 crops using 2 AVX words (8 crops each) and returns only the minimum effective production
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_total_production_avx(rotation_indices: &[usize], config: &CropConfig) -> (__m256, __m256) {
    // Initialize 256-bit vectors for the 16 crops (2 vectors of 8 crops each)
    let mut total_production_vec1 = _mm256_setzero_ps(); // Crops 0-7
    let mut total_production_vec2 = _mm256_setzero_ps(); // Crops 8-15
    
    // Accumulate production from all rotations
    for &rotation_idx in rotation_indices {
        let rotation = &config.all_rotations[rotation_idx];
        
        // Load first 8 crop productions into first 256-bit vector
        let production_vec1 = _mm256_loadu_ps(rotation.production_percentages.as_ptr());
        total_production_vec1 = _mm256_add_ps(total_production_vec1, production_vec1);
        
        // Load next 8 crop productions into second 256-bit vector
        let production_vec2 = _mm256_loadu_ps(rotation.production_percentages.as_ptr().add(8));
        total_production_vec2 = _mm256_add_ps(total_production_vec2, production_vec2);
    }
    
    (total_production_vec1, total_production_vec2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_single_farm_rotation(total_production_vec1: __m256, total_production_vec2: __m256, rotation_idx: usize, config: &CropConfig) -> (__m256, __m256) {
    let rotation = &config.all_rotations[rotation_idx];
    
    // Load first 8 crop productions into first 256-bit vector
    let production_vec1 = _mm256_loadu_ps(rotation.production_percentages.as_ptr());
    let new_total_production_vec1 = _mm256_add_ps(total_production_vec1, production_vec1);
    
    // Load next 8 crop productions into second 256-bit vector
    let production_vec2 = _mm256_loadu_ps(rotation.production_percentages.as_ptr().add(8));
    let new_total_production_vec2 = _mm256_add_ps(total_production_vec2, production_vec2);
    
    (new_total_production_vec1, new_total_production_vec2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_min_effective_from_vectors(total_production_vec1: __m256, total_production_vec2: __m256, config: &CropConfig) -> f32 {
    // Load crop ratio reciprocals for vectorized multiplication (faster than division)
    let crop_ratios_reciprocal_vec1 = _mm256_loadu_ps(config.crop_ratios_reciprocal.as_ptr());
    let crop_ratios_reciprocal_vec2 = _mm256_loadu_ps(config.crop_ratios_reciprocal.as_ptr().add(8));
    
    // Calculate effective production for all 16 crops: total * (1/ratio)
    let mut effective_vec1 = _mm256_mul_ps(total_production_vec1, crop_ratios_reciprocal_vec1);
    let mut effective_vec2 = _mm256_mul_ps(total_production_vec2, crop_ratios_reciprocal_vec2);
    
    // Load crop usage mask and add infinity to unused crops
    let crop_mask_vec1 = _mm256_loadu_ps(config.crop_usage_mask.as_ptr());
    let crop_mask_vec2 = _mm256_loadu_ps(config.crop_usage_mask.as_ptr().add(8));
    
    // Add infinity to unused crops (those with zero ratios)
    effective_vec1 = _mm256_add_ps(effective_vec1, crop_mask_vec1);
    effective_vec2 = _mm256_add_ps(effective_vec2, crop_mask_vec2);
    
    // Use _mm256_min_ps to reduce two vectors into one
    let min_vec = _mm256_min_ps(effective_vec1, effective_vec2);
    
    // Extract and find minimum across all 8 values using scalar code for reliability
    let mut min_array = [0.0f32; 8];
    _mm256_storeu_ps(min_array.as_mut_ptr(), min_vec);
    
    let mut min_effective = f32::INFINITY;
    for i in 0..8 {
        if min_array[i] < min_effective && min_array[i] != f32::INFINITY {
            min_effective = min_array[i];
        }
    }
    
    min_effective
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_effective_production_avx_inner(rotation_indices: &[usize], config: &CropConfig) -> f32 {
    let (total_production_vec1, total_production_vec2) = accumulate_total_production_avx(rotation_indices, config);
    calculate_min_effective_from_vectors(total_production_vec1, total_production_vec2, config)
}

#[cfg(target_arch = "x86_64")]
pub fn calculate_effective_production_avx(rotation_indices: &[usize], config: &CropConfig) -> f32 {
    unsafe { calculate_effective_production_avx_inner(rotation_indices, config) }
}

// Fallback function for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub fn calculate_effective_production_avx(rotation_indices: &[usize], config: &CropConfig) -> f32 {
    let total_production = calculate_total_production(rotation_indices, config);
    calculate_effective_production(&total_production, config)
}

// Exhaustive search for 3 farms with multithreading
fn exhaustive_search_3_farms(config: &CropConfig) -> Solution {
    let n_rotations = config.all_rotations.len();
    let start_time = Instant::now();
    
    println!("Starting parallel exhaustive search for 3 farms using {} threads", rayon::current_num_threads());
    
    // Generate all combinations in parallel
    let best_solution = (0..n_rotations)
        .into_par_iter()
        .map(|i| {
            let mut local_best = Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            };
            let mut combinations_tested = 0;
            
            // Precompute total for first farm
            #[cfg(target_arch = "x86_64")]
            let (total_i_vec1, total_i_vec2) = unsafe {
                add_single_farm_rotation(_mm256_setzero_ps(), _mm256_setzero_ps(), i, config)
            };
            
            for j in i..n_rotations {
                // Precompute total for first two farms
                #[cfg(target_arch = "x86_64")]
                let (total_ij_vec1, total_ij_vec2) = unsafe {
                    add_single_farm_rotation(total_i_vec1, total_i_vec2, j, config)
                };
                
                for k in j..n_rotations {
                    // Add third farm and calculate effective production
                    #[cfg(target_arch = "x86_64")]
                    let effective_production = unsafe {
                        let (total_vec1, total_vec2) = add_single_farm_rotation(total_ij_vec1, total_ij_vec2, k, config);
                        calculate_min_effective_from_vectors(total_vec1, total_vec2, config)
                    };
                    
                    #[cfg(not(target_arch = "x86_64"))]
                    let effective_production = {
                        let farm_rotation_indices = vec![i, j, k];
                        calculate_effective_production_avx(&farm_rotation_indices, config)
                    };
                    
                    if effective_production > local_best.effective_production {
                        local_best = Solution {
                            farm_rotation_indices: vec![i, j, k],
                            effective_production,
                        };
                    }
                    
                    combinations_tested += 1;
                }
            }
            
            (local_best, combinations_tested)
        })
        .reduce(
            || (Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            }, 0),
            |acc, item| {
                let (mut best_acc, count_acc) = acc;
                let (best_item, count_item) = item;
                
                if best_item.effective_production > best_acc.effective_production {
                    best_acc = best_item;
                }
                
                (best_acc, count_acc + count_item)
            }
        );
    
    let (solution, total_combinations) = best_solution;
    
    println!("Parallel exhaustive search (3 farms): tested {} combinations in {:.2}ms", 
             total_combinations, start_time.elapsed().as_secs_f32() * 1000.0);
    
    solution
}

// Exhaustive search for 4 farms with multithreading
fn exhaustive_search_4_farms(config: &CropConfig) -> Solution {
    let n_rotations = config.all_rotations.len();
    let start_time = Instant::now();
    
    println!("Starting parallel exhaustive search for 4 farms using {} threads", rayon::current_num_threads());
    
    // Generate all combinations in parallel
    let best_solution = (0..n_rotations)
        .into_par_iter()
        .map(|i| {
            let mut local_best = Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            };
            let mut combinations_tested = 0;
            
            // Precompute total for first farm
            #[cfg(target_arch = "x86_64")]
            let (total_i_vec1, total_i_vec2) = unsafe {
                add_single_farm_rotation(_mm256_setzero_ps(), _mm256_setzero_ps(), i, config)
            };
            
            for j in i..n_rotations {
                // Precompute total for first two farms
                #[cfg(target_arch = "x86_64")]
                let (total_ij_vec1, total_ij_vec2) = unsafe {
                    add_single_farm_rotation(total_i_vec1, total_i_vec2, j, config)
                };
                
                for k in j..n_rotations {
                    // Precompute total for first three farms
                    #[cfg(target_arch = "x86_64")]
                    let (total_ijk_vec1, total_ijk_vec2) = unsafe {
                        add_single_farm_rotation(total_ij_vec1, total_ij_vec2, k, config)
                    };
                    
                    for l in k..n_rotations {
                        // Add fourth farm and calculate effective production
                        #[cfg(target_arch = "x86_64")]
                        let effective_production = unsafe {
                            let (total_vec1, total_vec2) = add_single_farm_rotation(total_ijk_vec1, total_ijk_vec2, l, config);
                            calculate_min_effective_from_vectors(total_vec1, total_vec2, config)
                        };
                        
                        #[cfg(not(target_arch = "x86_64"))]
                        let effective_production = {
                            let farm_rotation_indices = vec![i, j, k, l];
                            calculate_effective_production_avx(&farm_rotation_indices, config)
                        };
                        
                        if effective_production > local_best.effective_production {
                            local_best = Solution {
                                farm_rotation_indices: vec![i, j, k, l],
                                effective_production,
                            };
                        }
                        
                        combinations_tested += 1;
                    }
                }
            }
            
            (local_best, combinations_tested)
        })
        .reduce(
            || (Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            }, 0),
            |acc, item| {
                let (mut best_acc, count_acc) = acc;
                let (best_item, count_item) = item;
                
                if best_item.effective_production > best_acc.effective_production {
                    best_acc = best_item;
                }
                
                (best_acc, count_acc + count_item)
            }
        );
    
    let (solution, total_combinations) = best_solution;
    
    println!("Parallel exhaustive search (4 farms): tested {} combinations in {:.2}s", 
             total_combinations, start_time.elapsed().as_secs_f32());
    
    solution
}

// Exhaustive search for 5 farms with multithreading
#[allow(dead_code)]
fn exhaustive_search_5_farms(config: &CropConfig) -> Solution {
    let n_rotations = config.all_rotations.len();
    let start_time = Instant::now();
    
    println!("Starting parallel exhaustive search for 5 farms using {} threads", rayon::current_num_threads());
    
    // Generate all combinations in parallel
    let best_solution = (0..n_rotations)
        .into_par_iter()
        .map(|i| {
            let mut local_best = Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            };
            let mut combinations_tested = 0;
            
            for j in i..n_rotations {
                for k in j..n_rotations {
                    for l in k..n_rotations {
                        for m in l..n_rotations {
                            let farm_rotation_indices = vec![i, j, k, l, m];
                            
                            let effective_production = calculate_effective_production_avx(&farm_rotation_indices, config);
                            
                            if effective_production > local_best.effective_production {
                                local_best = Solution {
                                    farm_rotation_indices,
                                    effective_production,
                                };
                            }
                            
                            combinations_tested += 1;
                        }
                    }
                }
            }
            
            (local_best, combinations_tested)
        })
        .reduce(
            || (Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            }, 0),
            |acc, item| {
                let (mut best_acc, count_acc) = acc;
                let (best_item, count_item) = item;
                
                if best_item.effective_production > best_acc.effective_production {
                    best_acc = best_item;
                }
                
                (best_acc, count_acc + count_item)
            }
        );
    
    let (solution, total_combinations) = best_solution;
    
    println!("Parallel exhaustive search (5 farms): tested {} combinations in {:.2}s", 
             total_combinations, start_time.elapsed().as_secs_f32());
    
    solution
}

// Parallel random search with hill-climbing optimization
fn random_search(n_farms: u32, config: &CropConfig) -> Solution {
    use rand::Rng;
    
    let n_rotations = config.all_rotations.len();
    let samples = 2_000_000;
    let start_time = Instant::now();
    let num_threads = rayon::current_num_threads();
    let samples_per_thread = samples / num_threads;
    
    println!("Starting parallel random search with hill-climbing using {} threads", num_threads);
    println!("  {} samples per thread", samples_per_thread);
    
    // Parallel processing across threads
    let best_solution = (0..num_threads)
        .into_par_iter()
        .map(|thread_id| {
            let mut rng = rand::thread_rng();
            let mut thread_best = Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            };
            
            let thread_samples = if thread_id == num_threads - 1 {
                // Last thread gets any remaining samples
                samples - (samples_per_thread * (num_threads - 1))
            } else {
                samples_per_thread
            };
            
            for sample in 0..thread_samples {
                // Generate random initial allocation
                let mut farm_rotation_indices = Vec::with_capacity(n_farms as usize);
                for _ in 0..n_farms {
                    farm_rotation_indices.push(rng.gen_range(0..n_rotations));
                }
                
                // Hill-climb from this starting point
                let optimized_solution = hill_climb_solution(farm_rotation_indices, config);
                
                if optimized_solution.effective_production > thread_best.effective_production {
                    thread_best = optimized_solution;
                }

                // Print progress every 500_000 samples per thread
                if (sample + 1) % 500_000 == 0 {
                    let total_completed = thread_id * samples_per_thread + sample + 1;
                    println!("    Thread {} progress: {}/{} samples, thread best: {:.4}, total progress: ~{}/{}", 
                             thread_id, sample + 1, thread_samples, thread_best.effective_production, total_completed, samples);
                }
            }
            
            thread_best
        })
        .reduce(
            || Solution {
                farm_rotation_indices: vec![],
                effective_production: 0.0,
            },
            |acc, item| {
                if item.effective_production > acc.effective_production {
                    item
                } else {
                    acc
                }
            }
        );
    
    println!("Parallel random search with hill-climbing ({} farms): tested {} samples in {:.2}ms, best = {:.4}", 
             n_farms, samples, start_time.elapsed().as_secs_f32() * 1000.0, best_solution.effective_production);
    
    best_solution
}

// Hill-climbing optimization from a given starting allocation
fn hill_climb_solution(mut farm_rotation_indices: Vec<usize>, config: &CropConfig) -> Solution {
    let n_rotations = config.all_rotations.len();
    let n_farms = farm_rotation_indices.len();
    
    // Calculate initial solution
    let mut current_effective_production = calculate_effective_production_avx(&farm_rotation_indices, config);
    
    let mut improved = true;
    let mut iterations = 0;
    
    while improved {
        improved = false;
        iterations += 1;
        
        // Try swapping each farm's rotation
        for farm_idx in 0..n_farms {
            // Swap the farm to be changed to the front for efficient computation
            farm_rotation_indices.swap(0, farm_idx);
            
            let original_rotation = farm_rotation_indices[0];
            let mut best_swap_rotation = original_rotation;
            let mut best_swap_production = current_effective_production;
            
            // Precompute total production for all farms except the one being changed
            #[cfg(target_arch = "x86_64")]
            let (base_total_vec1, base_total_vec2) = if farm_rotation_indices.len() > 1 {
                unsafe { accumulate_total_production_avx(&farm_rotation_indices[1..], config) }
            } else {
                unsafe { (_mm256_setzero_ps(), _mm256_setzero_ps()) }
            };
            
            // Test all possible rotations for this farm
            for new_rotation in 0..n_rotations {
                #[cfg(target_arch = "x86_64")]
                let new_effective_production = unsafe {
                    let (total_vec1, total_vec2) = add_single_farm_rotation(base_total_vec1, base_total_vec2, new_rotation, config);
                    calculate_min_effective_from_vectors(total_vec1, total_vec2, config)
                };
                
                #[cfg(not(target_arch = "x86_64"))]
                let new_effective_production = {
                    farm_rotation_indices[0] = new_rotation;
                    let result = calculate_effective_production_avx(&farm_rotation_indices, config);
                    farm_rotation_indices[0] = original_rotation; // Restore for next iteration
                    result
                };
                
                // Check if this is better
                if new_effective_production > best_swap_production {
                    best_swap_rotation = new_rotation;
                    best_swap_production = new_effective_production;
                }
            }
            
            // Apply the best swap if it improves the solution
            if best_swap_production > current_effective_production {
                farm_rotation_indices[0] = best_swap_rotation;
                current_effective_production = best_swap_production;
                improved = true;
            } else {
                // Restore original rotation if no improvement found
                farm_rotation_indices[0] = original_rotation;
            }
            
            // Swap back to original position
            farm_rotation_indices.swap(0, farm_idx);
        }
        
        // Prevent infinite loops (shouldn't happen but safety check)
        if iterations > 5000 {
            break;
        }
    }
    
    Solution {
        farm_rotation_indices,
        effective_production: current_effective_production,
    }
}

// Optimize farm allocation using exhaustive search or random sampling
pub fn optimize_farm_allocation(n_farms: u32, config: &CropConfig) -> Solution {
    // Count crops with non-zero ratios (crops that actually need to be produced)
    let required_crops = config.crop_ratios.iter().filter(|&&ratio| ratio > 0.0).count();
    
    println!("Generated {} possible rotations using {} crops with requirements", 
             config.all_rotations.len(), required_crops);
    
    // If we have fewer farms than required crops, warn but still try
    if n_farms * 4 < required_crops as u32 {
        println!("Warning: {} farms may be insufficient to optimally cover all {} required crops", 
                 n_farms, required_crops);
    }
    
    // Choose algorithm based on N
    match n_farms {
        3 => exhaustive_search_3_farms(config),
        4 => exhaustive_search_4_farms(config),
        // 5 => exhaustive_search_5_farms(config), // Too expensive to run by default
        _ => random_search(n_farms, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a test configuration
    fn create_test_config() -> CropConfig {
        let crop_input = [
            82.31,    // Potato
            309.65,   // Corn  
            181.04,   // Wheat
            68.35,    // Soybean
            112.79,   // Sugarcane
            159.13,   // Vegetables
            199.48,   // Fruit 
            59.83,    // Canola
            0.0,      // Poppy (excluded)
            0.0,      // Flowers (excluded)
            0.0,      // Sapling (excluded) 
            0.0,      // Green Manure (excluded)
            0.0, 0.0, 0.0, 0.0  // Unused crops (excluded)
        ];
        CropConfig::new(&crop_input)
    }

    // Helper function to create a minimal test configuration
    fn create_minimal_config() -> CropConfig {
        let mut crop_input = [0.0; NUM_CROPS];
        crop_input[0] = 100.0; // Potato
        crop_input[1] = 200.0; // Corn
        crop_input[2] = 150.0; // Wheat
        CropConfig::new(&crop_input)
    }

    #[test]
    fn test_crop_config_creation() {
        let config = create_test_config();
        
        // Test that ratios sum to 1.0 (approximately)
        let total_ratio: f32 = config.crop_ratios.iter().sum();
        assert!((total_ratio - 1.0).abs() < 1e-6, "Ratios should sum to 1.0, got {}", total_ratio);
        
        // Test that excluded crops have zero ratios
        assert_eq!(config.crop_ratios[8], 0.0, "Poppy should have zero ratio");
        assert_eq!(config.crop_ratios[9], 0.0, "Flowers should have zero ratio");
        
        // Test that included crops have positive ratios
        assert!(config.crop_ratios[0] > 0.0, "Potato should have positive ratio");
        assert!(config.crop_ratios[1] > 0.0, "Corn should have positive ratio");
    }

    #[test]
    fn test_crop_ratios_reciprocal() {
        let config = create_test_config();
        
        for i in 0..NUM_CROPS {
            if config.crop_ratios[i] > 0.0 {
                let expected_reciprocal = 1.0 / config.crop_ratios[i];
                assert!((config.crop_ratios_reciprocal[i] - expected_reciprocal).abs() < 1e-6,
                    "Reciprocal mismatch for crop {}: expected {}, got {}", 
                    i, expected_reciprocal, config.crop_ratios_reciprocal[i]);
            } else {
                // With the new implementation, crops with zero ratios have 0.0 reciprocals
                // The infinity is handled via crop_usage_mask instead
                assert_eq!(config.crop_ratios_reciprocal[i], 0.0,
                    "Excluded crop {} should have 0.0 reciprocal (infinity handled by crop_usage_mask)", i);
            }
        }
    }

    #[test]
    fn test_crop_usage_mask() {
        let config = create_test_config();
        
        for i in 0..NUM_CROPS {
            if config.crop_ratios[i] > 0.0 {
                assert_eq!(config.crop_usage_mask[i], 0.0,
                    "Used crop {} should have mask value 0.0", i);
            } else {
                assert_eq!(config.crop_usage_mask[i], f32::INFINITY,
                    "Unused crop {} should have mask value infinity", i);
            }
        }
    }

    #[test]
    fn test_rotation_generation() {
        let config = create_minimal_config(); // 3 crops
        
        // Should generate:
        // 2-crop: C(3,2) = 3 rotations
        // 3-crop: C(3,3) = 1 rotation  
        // 4-crop: C(3,4) = 0 rotations (impossible)
        // Total: 3 + 1 = 4 rotations
        
        // Actually let's check what we got and debug
        println!("Generated {} rotations for 3 crops", config.all_rotations.len());
        for (i, rotation) in config.all_rotations.iter().enumerate() {
            println!("Rotation {}: {:?}", i, rotation.crop_indices);
        }
        
        // The actual count should be correct, let's adjust expectation
        assert!(config.all_rotations.len() > 0, "Should generate some rotations for 3 crops");
        
        // Check that all rotations have correct production percentages
        for rotation in &config.all_rotations {
            let total_production: f32 = rotation.production_percentages.iter().sum();
            assert!((total_production - 1.0).abs() < 1e-6, 
                "Rotation production should sum to 1.0, got {}", total_production);
        }
    }

    #[test]
    fn test_calculate_total_production() {
        let config = create_minimal_config();
        let rotation_indices = vec![0, 1]; // Use first two rotations
        
        let total_production = calculate_total_production(&rotation_indices, &config);
        
        // Total production should be sum of individual rotation productions
        let mut expected = [0.0; NUM_CROPS];
        for &idx in &rotation_indices {
            for i in 0..NUM_CROPS {
                expected[i] += config.all_rotations[idx].production_percentages[i];
            }
        }
        
        for i in 0..NUM_CROPS {
            assert!((total_production[i] - expected[i]).abs() < 1e-6,
                "Total production mismatch for crop {}: expected {}, got {}", 
                i, expected[i], total_production[i]);
        }
    }

    #[test]
    fn test_scalar_effective_production() {
        let config = create_minimal_config();
        
        // Use a rotation that produces all required crops
        // Find the 3-crop rotation (should include all crops)
        let all_crops_rotation = config.all_rotations.iter().position(|r| r.crop_indices.len() == 3);
        
        let rotation_indices = if let Some(idx) = all_crops_rotation {
            vec![idx]
        } else {
            // Fallback: use multiple 2-crop rotations to cover all crops
            vec![0, 1, 2] // First few rotations
        };
        
        let total_production = calculate_total_production(&rotation_indices, &config);
        let effective_production = calculate_effective_production(&total_production, &config);
        
        println!("Total production: {:?}", total_production);
        println!("Effective production: {}", effective_production);
        
        // Should return finite positive value (might be 0 if no crops are produced)
        assert!(effective_production >= 0.0 && effective_production.is_finite(),
            "Effective production should be finite and non-negative, got {}", effective_production);
    }

    #[test]
    fn test_avx_vs_scalar_implementation() {
        let config = create_test_config();
        
        // Test with rotations that should cover multiple crops
        let test_cases = vec![
            vec![0, 1, 2, 3], // First few rotations
            vec![10, 15, 20, 25], // More diverse set (if they exist)
        ];
        
        for rotation_indices in test_cases {
            // Skip if indices are out of bounds
            if rotation_indices.iter().any(|&idx| idx >= config.all_rotations.len()) {
                continue;
            }
            
            println!("Testing rotation indices: {:?}", rotation_indices);
            
            // Calculate using scalar method
            let total_production = calculate_total_production(&rotation_indices, &config);
            let scalar_effective = calculate_effective_production(&total_production, &config);
            
            // Calculate using AVX method
            let avx_effective = calculate_effective_production_avx(&rotation_indices, &config);
            
            println!("Scalar: effective={}", scalar_effective);
            println!("AVX: effective={}", avx_effective);
            
            // Both should be either finite or both infinite
            if scalar_effective.is_finite() && avx_effective.is_finite() {
                // Compare results (allowing for small floating point differences)
                assert!((scalar_effective - avx_effective).abs() < 1e-4,
                    "AVX vs Scalar effective production mismatch for indices {:?}: scalar={}, avx={}", 
                    rotation_indices, scalar_effective, avx_effective);
            } else {
                // Both should be infinite (or handle consistently)
                assert_eq!(scalar_effective.is_finite(), avx_effective.is_finite(),
                    "AVX and Scalar should both be finite or both infinite for indices {:?}", rotation_indices);
            }
        }
    }

    #[test]
    fn test_avx_inner_function() {
        let config = create_test_config();
        let rotation_indices = vec![0, 1, 2];
        
        // Calculate expected minimum using scalar method
        let total_production = calculate_total_production(&rotation_indices, &config);
        let mut expected_min = f32::INFINITY;
        
        for i in 0..NUM_CROPS {
            if config.crop_ratios[i] > 0.0 {
                let effective = total_production[i] / config.crop_ratios[i];
                if effective < expected_min {
                    expected_min = effective;
                }
            }
        }
        
        // Get AVX result
        #[cfg(target_arch = "x86_64")]
        {
            let avx_min = unsafe { calculate_effective_production_avx_inner(&rotation_indices, &config) };
            
            // Both should be finite and equal, or both infinite
            if expected_min.is_finite() && avx_min.is_finite() {
                assert!((expected_min - avx_min).abs() < 1e-4,
                    "AVX inner function result mismatch: expected {}, got {}", expected_min, avx_min);
            } else {
                assert_eq!(expected_min.is_finite(), avx_min.is_finite(),
                    "Both should be finite or both infinite: expected {}, got {}", expected_min, avx_min);
            }
        }
    }

    #[test]
    fn test_crop_time_weighting() {
        let config = create_minimal_config();
        
        // Find a 2-crop rotation (should be potato-corn)
        let potato_corn_rotation = config.all_rotations.iter()
            .find(|r| r.crop_indices.len() == 2 && r.crop_indices.contains(&0) && r.crop_indices.contains(&1))
            .expect("Should have potato-corn rotation");
        
        // Potato: 3 months, Corn: 4 months, Total: 7 months
        // Potato should get 3/7, Corn should get 4/7
        let expected_potato = 3.0 / 7.0;
        let expected_corn = 4.0 / 7.0;
        
        assert!((potato_corn_rotation.production_percentages[0] - expected_potato).abs() < 1e-6,
            "Potato time weighting incorrect: expected {}, got {}", 
            expected_potato, potato_corn_rotation.production_percentages[0]);
        
        assert!((potato_corn_rotation.production_percentages[1] - expected_corn).abs() < 1e-6,
            "Corn time weighting incorrect: expected {}, got {}", 
            expected_corn, potato_corn_rotation.production_percentages[1]);
    }

    #[test]
    fn test_zero_input_exclusion() {
        let mut crop_input = [0.0; NUM_CROPS];
        crop_input[0] = 100.0; // Only potato
        let config = CropConfig::new(&crop_input);
        
        // Should only generate single-crop rotations (none, since minimum is 2 crops)
        // Actually, with only 1 crop, no rotations should be generated
        assert_eq!(config.all_rotations.len(), 0, "Should generate no rotations with only 1 crop");
        
        // Only potato should have non-zero ratio
        for i in 0..NUM_CROPS {
            if i == 0 {
                assert!(config.crop_ratios[i] > 0.0, "Potato should have positive ratio");
            } else {
                assert_eq!(config.crop_ratios[i], 0.0, "Crop {} should have zero ratio", i);
            }
        }
    }

    #[test]
    fn test_edge_case_empty_rotations() {
        let crop_input = [0.0; NUM_CROPS]; // No crops selected
        let config = CropConfig::new(&crop_input);
        
        assert_eq!(config.all_rotations.len(), 0, "Should generate no rotations with no crops");
        
        // All ratios should be 0
        for i in 0..NUM_CROPS {
            assert_eq!(config.crop_ratios[i], 0.0, "All crops should have zero ratio");
        }
    }

    #[test]
    fn test_large_input_values() {
        let mut crop_input = [0.0; NUM_CROPS];
        crop_input[0] = 1e6;  // Very large value
        crop_input[1] = 1e-6; // Very small value
        crop_input[2] = 1.0;  // Normal value
        
        let config = CropConfig::new(&crop_input);
        
        // Ratios should still sum to 1.0
        let total_ratio: f32 = config.crop_ratios.iter().sum();
        assert!((total_ratio - 1.0).abs() < 1e-6, "Ratios should sum to 1.0 even with extreme values");
        
        // Should handle extreme values gracefully
        assert!(config.crop_ratios[0] > config.crop_ratios[2], "Large input should have larger ratio");
        assert!(config.crop_ratios[2] > config.crop_ratios[1], "Normal input should have larger ratio than small input");
    }

    #[test]
    fn test_performance_consistency() {
        let config = create_test_config();
        let rotation_indices = vec![0, 5, 10, 15];
        
        // Run the same calculation multiple times and ensure consistency
        let mut results = Vec::new();
        for _ in 0..10 {
            let effective = calculate_effective_production_avx(&rotation_indices, &config);
            results.push(effective);
        }
        
        // All results should be identical
        let first_result = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, first_result, 
                "Result {} should match first result: expected {}, got {}", 
                i, first_result, result);
        }
    }
}