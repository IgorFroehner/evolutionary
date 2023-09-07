use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, PathElement},
    series::LineSeries,
    style::{Color, BLACK, BLUE, RED, WHITE},
};

pub struct Metrics {
    pub test_name: String,
    pub best_fitnesses: Vec<f64>,
    pub avg_fitnesses: Vec<f64>,
    pub iterations: u32,
}

impl Metrics {
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            best_fitnesses: Vec::new(),
            avg_fitnesses: Vec::new(),
            iterations: 0,
        }
    }

    pub fn record(&mut self, best_fitness: f64, avg_fitness: f64) {
        self.best_fitnesses.push(best_fitness);
        self.avg_fitnesses.push(avg_fitness);
        self.iterations += 1;
    }

    pub fn plot_chart(&self) -> Result<(), Box<dyn std::error::Error>> {
        let file_name = format!("{}.png", self.test_name);

        let root = BitMapBackend::new(&file_name, (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_y = self
            .best_fitnesses
            .iter()
            .chain(self.avg_fitnesses.iter())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("{}", self.test_name), ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0f64..self.best_fitnesses.len() as f64, 0f64..(*max_y + 0.1 * *max_y))?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                self.best_fitnesses
                    .iter()
                    .enumerate()
                    .map(|(x, y)| (x as f64, *y)),
                &RED,
            ))?
            .label("Best Fitness")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .draw_series(LineSeries::new(
                self.avg_fitnesses
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
}
