use plotters::prelude::*;
use plotters::style::{RED, WHITE};

pub fn convert_bin(vec: &Vec<bool>) -> f64 {
    let mut res = 0.0;
    for i in 0..vec.len() {
        res += (if vec[i] { 1.0 } else { 0.0 }) * 2f64.powf(i as f64);
    }

    res
}

pub fn within_range(range: (f64, f64), l: f64, d: f64) -> f64 {
    let a = range.1 - range.0;
    let b = 2f64.powf(l) - 1.;

    (range.0 + (a / b) * d).floor()
}

pub fn plot_chart(
    best_fitness: &Vec<f64>,
    average_fitness: &Vec<f64>,
    path: &String,
    test_name: &String,
) -> Result<(), Box<dyn std::error::Error>> {
    let max = *average_fitness
        .iter()
        .chain(best_fitness)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0);

    let normalized_avg = normalize(&average_fitness, max);
    let normalized_best = normalize(&best_fitness, max);

    let min = normalized_avg
        .iter()
        .chain(&normalized_best)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    let y_axis_start = if *min > 0.5 { min } else { min };

    let y_axis_start = y_axis_start - (y_axis_start * 0.1);

    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{}", test_name), ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..normalized_best.len() as f64, (y_axis_start)..1.05)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            normalized_best
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, *y)),
            &RED,
        ))?
        .label("Best Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            normalized_avg
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, *y)),
            &BLUE,
        ))?
        .label("Average Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

pub fn plot_boxplot(
    quartiles: &Vec<Quartiles>,
    labels: &Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("boxplot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = quartiles
        .iter()
        .map(|q| q.values()[0])
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = quartiles
        .iter()
        .map(|q| q.values()[4])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("128 Queens Score", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(labels.into_segmented(), min..max)?;

    chart.configure_mesh().draw()?;

    for (quartile, label) in quartiles.iter().zip(labels.iter()) {
        chart.draw_series(vec![Boxplot::new_vertical(
            SegmentValue::CenterOf(label),
            quartile,
        )])?;
    }

    Ok(())
}

fn normalize(values: &Vec<f64>, max: f64) -> Vec<f64> {
    let min = 0.0;
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

#[cfg(test)]
mod test {
    use crate::utils::{convert_bin, within_range};

    #[test]
    fn convert_bin_test() {
        let seven_vec = vec![true, true, true, false]; // 1 + 2 + 4
        assert_eq!(convert_bin(&seven_vec), 7.0);

        let thirteen_vec: Vec<bool> = vec![true, false, true, true]; // 1 + 0 + 4 + 8
        assert_eq!(convert_bin(&thirteen_vec), 13.0);

        let sixty_four_vec: Vec<bool> = vec![false, false, false, false, false, false, true];
        assert_eq!(convert_bin(&sixty_four_vec), 64.0);
    }

    #[test]
    fn within_range_test_1() {
        let value_vec = vec![true, true, true, true, true, true, true, true, true, true];
        let value = convert_bin(&value_vec);

        let range = (0., 1.);
        let l = value_vec.len() as f64;
        let d = value;

        println!("value: {}", value);

        assert_eq!(within_range(range, l, d), range.1);
    }

    #[test]
    fn within_range_test_2() {
        let range = (0., 16.);
        let l = 5.0;
        let d = 30.0;

        assert_eq!(within_range(range, l, d), 15.0);
    }
}
