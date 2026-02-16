use anyhow::Result;

mod config;
mod errors;
mod logging;

fn main() -> Result<()> {
    let config = config::Config::load()?;
    logging::init_logging(&config);
    tracing::info!("memcp starting");
    Ok(())
}
