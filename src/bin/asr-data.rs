use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "asr-data", version, about = "ASR data utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert a FASR AudioList binary file to an ASR SQLite AudioDb.
    ConvertFasr(ConvertFasrArgs),
    /// Import a legacy ASR MessagePack file into a SQLite AudioDb.
    ImportMsgpack(ImportMsgpackArgs),
    /// Show summary information for an ASR SQLite AudioDb.
    Info(InfoArgs),
}

#[derive(Debug, Clone, Parser)]
struct ConvertFasrArgs {
    /// FASR AudioList binary file.
    #[arg(short = 'i', long, value_name = "PATH")]
    input: PathBuf,

    /// Output ASR SQLite AudioDb file.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: PathBuf,
}

#[derive(Debug, Clone, Parser)]
struct ImportMsgpackArgs {
    /// Legacy ASR MessagePack file.
    #[arg(short = 'i', long, value_name = "PATH")]
    input: PathBuf,

    /// Output ASR SQLite AudioDb file.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: PathBuf,
}

#[derive(Debug, Clone, Parser)]
struct InfoArgs {
    /// ASR SQLite AudioDb file.
    #[arg(value_name = "PATH")]
    input: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ConvertFasr(args) => {
            let summary = asr_data::convert_fasr_audiolist_to_db(&args.input, &args.output)?;
            eprintln!(
                "converted FASR AudioList | records={} input={} output={}",
                summary.records,
                args.input.display(),
                args.output.display()
            );
            Ok(())
        }
        Command::ImportMsgpack(args) => {
            let imported = asr_data::import_legacy_msgpack_to_db(&args.input, &args.output)?;
            eprintln!(
                "imported legacy MessagePack | audios={} input={} output={}",
                imported,
                args.input.display(),
                args.output.display()
            );
            Ok(())
        }
        Command::Info(args) => {
            let info = asr_data::read_audio_db_info(&args.input)?;
            eprintln!(
                "AudioDb | schema={} audios={} duration_ms={} path={}",
                info.schema_version,
                info.audios,
                info.total_duration.0,
                args.input.display()
            );
            Ok(())
        }
    }
}
