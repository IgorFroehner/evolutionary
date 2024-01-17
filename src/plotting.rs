use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::data::Quartiles;
use plotters::element::{Boxplot, PathElement};
use plotters::prelude::*;

fn normalize(values: &Vec<f64>, max: f64) -> Vec<f64> {
    let min = 0.0;
    values.iter().map(|&v| (v - min) / (max - min)).collect()
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
