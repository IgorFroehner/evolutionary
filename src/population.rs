use std::fmt::Debug;
use rand::prelude::*;

use serde_derive::Deserialize;

#[derive(Debug, Deserialize)]
pub enum GeneCod {
    Int,
    IntPerm,
    Bin,
    Real,
}


#[derive(Debug, Clone)]
pub struct Int(pub Vec<i64>, f64);

#[derive(Debug, Clone)]
pub struct IntPerm(pub Vec<i64>, f64);

#[derive(Debug, Clone)]
pub struct Bin(pub Vec<bool>, pub f64);

#[derive(Debug, Clone)]
pub struct Real(pub Vec<f64>, f64);

pub trait Individual: 'static + Clone + Send + Sync {
    type Gene: Debug + Send + Sync;
    type RangeType: Send + Sync;

    fn generate_member(dimension: u32, b: &Self::RangeType) -> Self;
    fn get_vec(&self) -> &Vec<Self::Gene>;
    fn set_fitness(&mut self, fitness: f64);
    fn get_fitness(&self) -> f64;
}

impl Individual for Int {
    type Gene = i64;
    type RangeType = (i64, i64);

    fn generate_member(dimension: u32, range: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self((0..dimension).map(|_| rng.gen_range(range.0..=range.1)).collect::<Vec<i64>>(), 0.0)
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }
}

impl Individual for IntPerm {
    type Gene = i64;
    type RangeType = ();

    fn generate_member(dimension: u32, _: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        let mut member = (1..=dimension as i64).collect::<Vec<i64>>();
        member.shuffle(&mut rng);

        Self(member, 0.0)
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }
}

impl Individual for Bin {
    type Gene = bool;
    type RangeType = ();

    fn generate_member(dimension: u32, _: &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self((0..dimension).map(|_| rng.gen()).collect::<Vec<bool>>(), 0.0)
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }
}

impl Individual for Real {
    type Gene = f64;
    type RangeType = (f64, f64);

    fn generate_member(dimension: u32, (a, b): &Self::RangeType) -> Self {
        let mut rng = thread_rng();

        Self((0..dimension).map(|_| rng.gen_range(*a..=*b)).collect::<Vec<f64>>(), 0.0)
    }

    fn get_vec(&self) -> &Vec<Self::Gene> {
        &self.0
    }

    fn get_fitness(&self) -> f64 {
        self.1
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.1 = fitness;
    }
}
