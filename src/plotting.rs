use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::data::Quartiles;
use plotters::element::{Boxplot, PathElement};
use plotters::prelude::*;

fn normalize(values: &[f64], max: f64) -> Vec<f64> {
    let min = 0.0;
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

pub fn plot_chart(
    best_fitness: &[f64],
    average_fitness: &[f64],
    path: &str,
    test_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let max = *average_fitness
        .iter()
        .chain(best_fitness)
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&1.0);

    let normalized_avg = normalize(average_fitness, max);
    let normalized_best = normalize(best_fitness, max);

    let min = normalized_avg
        .iter()
        .chain(&normalized_best)
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&0.0);

    let y_axis_start = min - min * 0.1;

    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(test_name, ("sans-serif", 40))
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
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            normalized_avg
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, *y)),
            &BLUE,
        ))?
        .label("Average Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    Ok(())
}

pub fn plot_boxplot(
    quartiles: &[Quartiles],
    labels: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("boxplot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let min = quartiles
        .iter()
        .map(|q| q.values()[0])
        .min_by(|a, b| a.total_cmp(b))
        .unwrap();
    let max = quartiles
        .iter()
        .map(|q| q.values()[4])
        .max_by(|a, b| a.total_cmp(b))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_scales_from_zero_to_max() {
        assert_eq!(normalize(&[0.0, 5.0, 10.0], 10.0), vec![0.0, 0.5, 1.0]);
        assert_eq!(normalize(&[-2.0, 4.0], 4.0), vec![-0.5, 1.0]);
    }

    #[test]
    fn plot_chart_saves_a_png() {
        let path = std::env::temp_dir().join(format!("evolutionary-{}-chart.png", std::process::id()));

        plot_chart(&[1.0, 2.0, 4.0], &[0.5, 1.0, 2.0], path.to_str().unwrap(), "Chart").unwrap();

        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(bytes.starts_with(b"\x89PNG"));
    }
}
