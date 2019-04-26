extern crate csv;
extern crate nalgebra;
extern crate serde;
extern crate statrs;

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{FisherSnedecor, StudentsT, Univariate};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;

type Record = HashMap<String, f64>;

#[derive(Debug)]
struct Extraction {
    headers: Vec<String>,
    rows: Vec<Record>,
}

#[derive(Debug)]
struct SummaryStats {
    regression_df: f64,
    residual_df: f64,
    total_df: f64,
    ssr: f64,
    sse: f64,
    sst: f64,
    msr: f64,
    mse: f64,
    f: f64,
    significance_f: f64,
    b: Vec<f64>,
    b_se: Vec<f64>,
    b_tstat: Vec<f64>,
    b_p: Vec<f64>,
    multiple_r: f64,
    r_square: f64,
    adj_r_square: f64,
    se: f64,
    observations: f64,
}

impl Extraction {
    fn new(headers: Vec<String>, rows: Vec<Record>) -> Extraction {
        // create new data set from csv
        Extraction { headers, rows }
    }

    fn summary(&self) -> SummaryStats {
        // Calculate summary statistics
        let (entries, y, rows_len, headers_len) = pre_process(&self.headers, &self.rows);
        let regression_df = (headers_len - 1) as f64;
        let residual_df = (rows_len - headers_len) as f64;
        let total_df = (rows_len - 1) as f64;
        let (b, lu) = coefficients(&self.headers, &self.rows, &y, &entries);
        let y_mean = mean(&y);
        let (ssr, sse, sst, msr, mse, f) = anova(&self.headers, &self.rows, &b, y_mean);
        let (b_se, b_tstat, b_p) = beta(mse, &lu, &b, residual_df);
        let r_square = ssr / sst;
        let multiple_r = r_square.sqrt() as f64;
        let adj_r_square = (1.0 - ((sse / residual_df) / (sst / total_df))) as f64;
        let se = ((sse / residual_df) / (sst / total_df)).sqrt() * (sst / total_df).sqrt() as f64;
        let observations = rows_len as f64;
        let significance_f = sig_f(regression_df, residual_df, f);

        // return summary statistics
        SummaryStats {
            regression_df,
            residual_df,
            total_df,
            ssr,
            sse,
            sst,
            msr,
            mse,
            f,
            significance_f,
            b,
            b_se,
            b_tstat,
            b_p,
            multiple_r,
            r_square,
            adj_r_square,
            se,
            observations,
        }
    }
}

impl fmt::Display for SummaryStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Format Display
        let title = "SUMMARY OUTPUT";

        let regression_statistics = format!(
            "{}\n{}\t{}\n{}\t{}\n{}\t{}\n{}\t{}\n{}\t{}\n",
            "Regression Statistics",
            "Multiple R:       ",
            self.multiple_r,
            "R Square:         ",
            self.r_square,
            "Adjusted R Square:",
            self.adj_r_square,
            "Standard Error:   ",
            self.se,
            "Observations:     ",
            self.observations
        );
        let anova_header = format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            "ANOVA      ",
            "df",
            "SS                    ",
            "MS                    ",
            "f                     ",
            "Significance F        "
        );
        let anova = format!(
            "{}\n{}\t{}\t{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\n{}\t{}\t{}\n",
            anova_header,
            "Regression:",
            self.regression_df,
            self.ssr,
            self.msr,
            self.f,
            self.significance_f,
            "Residual:  ",
            self.residual_df,
            self.sse,
            self.mse,
            "Total:     ",
            self.total_df,
            self.sst
        );

        let coef_header = format!(
            "{}\t{}\t{}\t{}\t{}",
            "  ",
            "Coefficient           ",
            "Standard Error        ",
            "t-Stat                ",
            "P-value               "
        );

        let coef = (0..)
            .zip(self.b.clone())
            .map(|(i, b)| {
                format!(
                    "{}{}\t{}\t{}\t{}\t{}",
                    "B", i, b, self.b_se[i], self.b_tstat[i], self.b_p[i],
                )
            })
            .collect::<Vec<String>>()
            .iter()
            .fold(String::new(), |mut s, n| {
                s = format!("{}\n{}", s, n);
                s
            });

        write!(
            f,
            "\n{}\n\n{}\n{}\n{}{}\n",
            title, regression_statistics, anova, coef_header, coef
        )
    }
}

fn sig_f(df1: f64, df2: f64, f: f64) -> f64 {
    // Significance F
    let f_dist =
        FisherSnedecor::new(df1, df2).expect("Failed to create Fisher-Snedecor distribution");
    1.0 - f_dist.cdf(f)
}

fn mean(y: &[f64]) -> f64 {
    // Calculates the mean of the dependent variable
    let mut y_mean: f64 = y.iter().sum();

    y_mean /= y.len() as f64;

    y_mean
}

fn anova(
    headers: &[String],
    rows: &[Record],
    b: &[f64],
    y_mean: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    // Analysis of Variance Table Calculations
    let ssr;
    let mut sse = 0.0;
    let mut sst = 0.0;

    for x in rows.iter() {
        let mut y_hat = 0.0;
        for (i, header) in headers.iter().enumerate() {
            if i == 0 {
                y_hat = b[i];
            } else {
                y_hat += b[i] * x[header];
            }
        }
        sse += (x[&headers[0]] - y_hat) * (x[&headers[0]] - y_hat);

        sst += (x[&headers[0]] - y_mean) * (x[&headers[0]] - y_mean);
    }

    ssr = sst - sse;

    // MSE = SSE / n - p - 1
    //     = (sum of (y - yhat)^2) / coefficient count including intercept

    let mse = sse / (rows.len() as f64 - headers.len() as f64);

    let msr = ssr / (headers.len() - 1) as f64;

    let f = msr / mse;

    (ssr, sse, sst, msr, mse, f)
}

fn beta(mse: f64, lu: &DMatrix<f64>, b: &[f64], df: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Calculate the standard error of coefficients;
    // where se = sqrt( diagonal values of matrix (MSE * (X'X)^-1));
    let se = (mse * lu).diagonal();

    let mut b_se: Vec<f64> = Vec::new();
    let mut b_tstat: Vec<f64> = Vec::new();
    let mut b_p: Vec<f64> = Vec::new();

    let students_t =
        StudentsT::new(0.0, 1.0, df).expect("Failed to create Student's T distribution");

    for (i, std_err) in se.iter().enumerate() {
        b_se.push(std_err.sqrt());
        b_tstat.push(b[i] / std_err.sqrt());
        b_p.push((1.0 - students_t.cdf((b[i] / std_err.sqrt()).abs())) * 2.0);
    }

    (b_se, b_tstat, b_p)
}

fn coefficients(
    headers: &[String],
    rows: &[Record],
    y: &[f64],
    entries: &[f64],
) -> (Vec<f64>, DMatrix<f64>) {
    // Calculate the Intercept and Coefficients (Bo... Bn)
    // where B =((X'X)^-1)X'y
    let independent_vars = DMatrix::from_row_slice(rows.len(), headers.len(), entries);

    let dependent_var = DVector::from_row_slice(y);

    let lu = (&independent_vars.transpose() * &independent_vars)
        .full_piv_lu()
        .try_inverse()
        .expect("Failed to calculate inverse.");

    let coefficients = &lu * &independent_vars.transpose() * dependent_var;

    let b = coefficients.iter().cloned().collect::<Vec<f64>>();

    (b, lu)
}

fn pre_process(headers: &[String], rows: &[Record]) -> (Vec<f64>, Vec<f64>, i64, i64) {
    // Create a seperate vector for dependent and independent variables
    // calculate the count of both rows and columns for future reference
    // so that they don't have to be recalculated throughout the regression
    let rows_len = rows.len() as i64;
    let headers_len = headers.len() as i64;
    let mut entries: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for x in rows.iter() {
        for (i, header) in headers.iter().enumerate() {
            if i == 0 {
                entries.push(1.0);
                y.push(x[header]);
            } else {
                entries.push(x[header]);
            }
        }
    }
    (entries, y, rows_len, headers_len)
}

fn csv_extractor() -> Result<Extraction, Box<Error>> {
    // Read in csv and return as Extraction struct
    let file_path = get_first_arg()?;

    let mut reader = csv::Reader::from_path(file_path)?;

    let headers = reader
        .headers()?
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<String>>();

    let mut entries: Vec<Record> = Vec::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        entries.push(record);
    }

    let export: Extraction = Extraction::new(headers, entries);

    Ok(export)
}

fn get_first_arg() -> Result<OsString, Box<Error>> {
    // Retrieve first argument from command line
    match env::args_os().nth(1) {
        None => Err(From::from("Expected 1 argument, but there were none.")),
        Some(file_path) => Ok(file_path),
    }
}

fn main() -> Result<(), Box<Error>> {
    //  Example usage: cargo run --release file.csv
    //  or: ./regression file_path.csv

    //  Notes:
    //  - the first column in the csv file will be read
    //  as the dependent variable. All other columns will be
    //  read as independent variables.
    //  - output is tab seperated so you can copy+paste into
    //  a spreadsheet

    let extraction = csv_extractor()?;

    let summary_stats = extraction.summary();

    println!("{}", &summary_stats);

    Ok(())
}
