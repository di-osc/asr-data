use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use serde::Deserialize;
use thiserror::Error;

use crate::db::{AudioDb, AudioDbError, AudioDbMode};

const README_FILE: &str = "README.md";
const TRAIN_DB_FILE: &str = "train.db";
const VAL_DB_FILE: &str = "val.db";
const TEST_DB_FILE: &str = "test.db";
const DEFAULT_MODELSCOPE_REVISION: &str = "master";
const MODELSCOPE_DATASET_FILES_URL: &str = "https://modelscope.cn/api/v1/datasets/<repo_id>/repo/tree?Recursive=True&Revision=<revision>&PageNumber=<page>&PageSize=<page_size>";
const MODELSCOPE_DATASET_PAGE_SIZE: usize = 200;
const MODELSCOPE_USER_AGENT: &str = "Mozilla/5.0 (compatible; asr-data/modelhub)";

type OptionalDatabase = Option<(PathBuf, AudioDb)>;

/// A named, versioned dataset with optional train, validation, and test
/// databases.
pub struct AudioDataset {
    name: String,
    version: String,
    license: String,
    snapshot_path: Option<PathBuf>,
    train_database_path: Option<PathBuf>,
    val_database_path: Option<PathBuf>,
    test_database_path: Option<PathBuf>,
    train: Option<AudioDb>,
    val: Option<AudioDb>,
    test: Option<AudioDb>,
}

#[derive(Debug, Error)]
pub enum AudioDatasetError {
    #[error("ModelScope repository id must not be empty")]
    EmptyRepositoryId,
    #[error("ModelScope revision must not be empty")]
    EmptyRevision,
    #[error("failed to download ModelScope dataset {repo_id:?} at revision {revision:?}: {source}")]
    ModelScopeDownload {
        repo_id: String,
        revision: String,
        #[source]
        source: anyhow::Error,
    },
    #[error("failed to read dataset license from {path:?}: {source}")]
    ReadLicense {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid dataset license metadata in {path:?}: {reason}")]
    InvalidLicense { path: PathBuf, reason: String },
    #[error("failed to open {split} database at {path:?}: {source}")]
    Database {
        split: &'static str,
        path: PathBuf,
        #[source]
        source: AudioDbError,
    },
}

impl AudioDataset {
    /// Wrap optional existing databases as an unnamed local dataset.
    ///
    /// This low-level constructor does not infer repository identity. Datasets
    /// loaded with [`Self::from_modelscope`] receive their name, version, and
    /// license from the repository and revision.
    pub fn new(train: OptionalDatabase, val: OptionalDatabase, test: OptionalDatabase) -> Self {
        let (train_database_path, train) = split_database_parts(train);
        let (val_database_path, val) = split_database_parts(val);
        let (test_database_path, test) = split_database_parts(test);
        Self {
            name: String::new(),
            version: String::new(),
            license: String::new(),
            snapshot_path: None,
            train_database_path,
            val_database_path,
            test_database_path,
            train,
            val,
            test,
        }
    }

    /// Download the complete ModelScope dataset repository with modelhub, then
    /// open any `train.db`, `val.db`, and `test.db` files that exist.
    ///
    /// Missing split databases are represented as [`None`]. Existing split
    /// databases are opened read-only. `cache_dir` defaults to
    /// [`modelhub::modelscope::cache_dir`], and `revision` defaults to
    /// `master`.
    pub async fn from_modelscope(
        repo_id: &str,
        revision: Option<&str>,
        cache_dir: Option<&Path>,
    ) -> Result<Self, AudioDatasetError> {
        Self::from_modelscope_with_downloader(&ModelHubDownloader, repo_id, revision, cache_dir)
            .await
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn snapshot_path(&self) -> Option<&Path> {
        self.snapshot_path.as_deref()
    }

    pub fn train_database_path(&self) -> Option<&Path> {
        self.train_database_path.as_deref()
    }

    pub fn val_database_path(&self) -> Option<&Path> {
        self.val_database_path.as_deref()
    }

    pub fn test_database_path(&self) -> Option<&Path> {
        self.test_database_path.as_deref()
    }

    pub fn train(&self) -> Option<&AudioDb> {
        self.train.as_ref()
    }

    pub fn val(&self) -> Option<&AudioDb> {
        self.val.as_ref()
    }

    pub fn test(&self) -> Option<&AudioDb> {
        self.test.as_ref()
    }

    pub fn into_databases(self) -> (Option<AudioDb>, Option<AudioDb>, Option<AudioDb>) {
        (self.train, self.val, self.test)
    }

    async fn from_modelscope_with_downloader<D: ModelScopeDownloader>(
        downloader: &D,
        repo_id: &str,
        revision: Option<&str>,
        cache_dir: Option<&Path>,
    ) -> Result<Self, AudioDatasetError> {
        if repo_id.trim().is_empty() {
            return Err(AudioDatasetError::EmptyRepositoryId);
        }
        let revision = revision.unwrap_or(DEFAULT_MODELSCOPE_REVISION);
        if revision.trim().is_empty() {
            return Err(AudioDatasetError::EmptyRevision);
        }
        let cache_dir = cache_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(modelhub::modelscope::cache_dir);

        let snapshot_path = downloader
            .download_dataset(repo_id, revision, &cache_dir)
            .await
            .map_err(|source| AudioDatasetError::ModelScopeDownload {
                repo_id: repo_id.to_owned(),
                revision: revision.to_owned(),
                source,
            })?;

        let license = read_optional_modelscope_license(&snapshot_path.join(README_FILE))?;
        let (train_database_path, train) =
            open_optional_database("train", &snapshot_path.join(TRAIN_DB_FILE))?;
        let (val_database_path, val) =
            open_optional_database("val", &snapshot_path.join(VAL_DB_FILE))?;
        let (test_database_path, test) =
            open_optional_database("test", &snapshot_path.join(TEST_DB_FILE))?;

        Ok(Self {
            name: repo_id.to_owned(),
            version: revision.to_owned(),
            license,
            snapshot_path: Some(snapshot_path),
            train_database_path,
            val_database_path,
            test_database_path,
            train,
            val,
            test,
        })
    }
}

impl fmt::Debug for AudioDataset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioDataset")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("license", &self.license)
            .field("snapshot_path", &self.snapshot_path)
            .field("train_database_path", &self.train_database_path)
            .field("val_database_path", &self.val_database_path)
            .field("test_database_path", &self.test_database_path)
            .finish_non_exhaustive()
    }
}

fn split_database_parts(database: OptionalDatabase) -> (Option<PathBuf>, Option<AudioDb>) {
    match database {
        Some((path, db)) => (Some(path), Some(db)),
        None => (None, None),
    }
}

fn open_optional_database(
    split: &'static str,
    database_path: &Path,
) -> Result<(Option<PathBuf>, Option<AudioDb>), AudioDatasetError> {
    if !database_path.exists() {
        return Ok((None, None));
    }
    let db = AudioDb::open(database_path, AudioDbMode::ReadOnly).map_err(|source| {
        AudioDatasetError::Database {
            split,
            path: database_path.to_path_buf(),
            source,
        }
    })?;
    Ok((Some(database_path.to_path_buf()), Some(db)))
}

fn read_optional_modelscope_license(readme_path: &Path) -> Result<String, AudioDatasetError> {
    if !readme_path.exists() {
        return Ok(String::new());
    }
    let readme =
        fs::read_to_string(readme_path).map_err(|source| AudioDatasetError::ReadLicense {
            path: readme_path.to_path_buf(),
            source,
        })?;
    parse_modelscope_license(&readme).map_err(|reason| AudioDatasetError::InvalidLicense {
        path: readme_path.to_path_buf(),
        reason,
    })
}

fn parse_modelscope_license(readme: &str) -> Result<String, String> {
    let readme = readme.strip_prefix('\u{feff}').unwrap_or(readme);
    let mut lines = readme.lines();
    if lines.next().map(str::trim) != Some("---") {
        return Ok(String::new());
    }

    let mut license = None;
    let mut closed = false;
    for line in lines {
        if line.trim() == "---" {
            closed = true;
            break;
        }
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "license" {
            continue;
        }
        if license.is_some() {
            return Err("front matter contains more than one license field".to_owned());
        }
        license = Some(parse_license_scalar(value.trim())?);
    }

    if !closed {
        return Err("README.md front matter is missing its closing delimiter".to_owned());
    }
    Ok(license.unwrap_or_default())
}

fn parse_license_scalar(value: &str) -> Result<String, String> {
    if value.is_empty() || value == "null" || value == "~" {
        return Ok(String::new());
    }
    if value.starts_with('"') {
        return serde_json::from_str(value)
            .map_err(|error| format!("license must be a valid quoted string: {error}"));
    }
    if let Some(value) = value.strip_prefix('\'') {
        let Some(value) = value.strip_suffix('\'') else {
            return Err("license must be a valid quoted string".to_owned());
        };
        return Ok(value.replace("''", "'"));
    }
    if matches!(value.as_bytes().first(), Some(b'[' | b'{')) {
        return Err("license must be a string".to_owned());
    }
    Ok(value.to_owned())
}

trait ModelScopeDownloader {
    async fn download_dataset(
        &self,
        repo_id: &str,
        revision: &str,
        cache_dir: &Path,
    ) -> anyhow::Result<PathBuf>;
}

struct ModelHubDownloader;

impl ModelScopeDownloader for ModelHubDownloader {
    async fn download_dataset(
        &self,
        repo_id: &str,
        revision: &str,
        cache_dir: &Path,
    ) -> anyhow::Result<PathBuf> {
        if let Err(error) =
            modelhub::modelscope::download_dataset_revision(repo_id, revision, cache_dir).await
        {
            let error_chain = format!("{error:#}");
            if !error_chain.contains("missing field `Success`") {
                return Err(error);
            }
            download_modelscope_snapshot_with_modelhub(repo_id, revision, cache_dir)
                .await
                .with_context(|| {
                    format!(
                        "modelhub whole-repository API was incompatible with the ModelScope response ({error_chain})"
                    )
                })?;
        }
        Ok(modelscope_snapshot_path(cache_dir, repo_id, revision))
    }
}

fn modelscope_snapshot_path(cache_dir: &Path, repo_id: &str, revision: &str) -> PathBuf {
    cache_dir
        .join("datasets")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join(revision)
}

#[derive(Deserialize)]
struct ModelScopeRepoTreeResponse {
    #[serde(rename = "Code")]
    code: i64,
    #[serde(rename = "Success")]
    success: Option<bool>,
    #[serde(rename = "Message", default)]
    message: String,
    #[serde(rename = "Data")]
    data: Option<ModelScopeRepoTreeData>,
}

#[derive(Deserialize)]
struct ModelScopeRepoTreeData {
    #[serde(rename = "Files")]
    files: Vec<ModelScopeRepoFile>,
}

#[derive(Deserialize)]
struct ModelScopeRepoFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Type", default)]
    file_type: String,
}

async fn download_modelscope_snapshot_with_modelhub(
    repo_id: &str,
    revision: &str,
    cache_dir: &Path,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let mut file_paths = Vec::new();

    for page in 1usize.. {
        let url = MODELSCOPE_DATASET_FILES_URL
            .replace("<repo_id>", repo_id)
            .replace("<revision>", &urlencoding::encode(revision))
            .replace("<page>", &page.to_string())
            .replace("<page_size>", &MODELSCOPE_DATASET_PAGE_SIZE.to_string());
        let response = client
            .get(url)
            .header("User-Agent", MODELSCOPE_USER_AGENT)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!(
                "failed to list ModelScope dataset files for {repo_id}@{revision}: HTTP {status}"
            );
        }
        let response = serde_json::from_str::<ModelScopeRepoTreeResponse>(&response.text().await?)?;
        if !response.success.unwrap_or(response.code == 200) {
            anyhow::bail!(
                "failed to list ModelScope dataset files for {repo_id}@{revision}: {}",
                response.message
            );
        }
        let files = response
            .data
            .context("ModelScope response did not include dataset file data")?
            .files;
        let count = files.len();
        file_paths.extend(
            files
                .into_iter()
                .filter(|file| file.file_type != "tree")
                .map(|file| file.path),
        );
        if count < MODELSCOPE_DATASET_PAGE_SIZE {
            break;
        }
    }

    for file_path in file_paths {
        modelhub::modelscope::download_dataset_file_revision(
            repo_id, &file_path, revision, cache_dir,
        )
        .await
        .with_context(|| format!("failed to download repository file {file_path:?}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DownloadRequest {
        repo_id: String,
        revision: String,
        cache_dir: PathBuf,
    }

    struct FakeDownloader {
        snapshot_path: PathBuf,
        request: Mutex<Option<DownloadRequest>>,
    }

    impl ModelScopeDownloader for FakeDownloader {
        async fn download_dataset(
            &self,
            repo_id: &str,
            revision: &str,
            cache_dir: &Path,
        ) -> anyhow::Result<PathBuf> {
            *self.request.lock().expect("request lock") = Some(DownloadRequest {
                repo_id: repo_id.to_owned(),
                revision: revision.to_owned(),
                cache_dir: cache_dir.to_path_buf(),
            });
            Ok(self.snapshot_path.clone())
        }
    }

    struct FailingDownloader;

    impl ModelScopeDownloader for FailingDownloader {
        async fn download_dataset(
            &self,
            _repo_id: &str,
            _revision: &str,
            _cache_dir: &Path,
        ) -> anyhow::Result<PathBuf> {
            anyhow::bail!("repository download failed")
        }
    }

    fn temp_snapshot(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("asr-data-{name}-{}", uuid::Uuid::new_v4()))
    }

    fn create_database(path: &Path) {
        drop(AudioDb::create(path).expect("create database"));
    }

    #[test]
    fn modelscope_downloads_snapshot_and_opens_only_present_split_databases() {
        let snapshot_path = temp_snapshot("modelscope-snapshot");
        fs::create_dir_all(&snapshot_path).expect("create snapshot");
        fs::write(
            snapshot_path.join(README_FILE),
            "---\nlicense: Apache License 2.0\nlanguage: zh\n---\n# Calls\n",
        )
        .expect("write README");
        create_database(&snapshot_path.join(TRAIN_DB_FILE));
        create_database(&snapshot_path.join(TEST_DB_FILE));

        let downloader = FakeDownloader {
            snapshot_path: snapshot_path.clone(),
            request: Mutex::new(None),
        };
        let cache_dir = std::env::temp_dir().join("asr-data-modelscope-cache");
        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        let dataset = runtime
            .block_on(AudioDataset::from_modelscope_with_downloader(
                &downloader,
                "di-osc/calls",
                Some("v1"),
                Some(&cache_dir),
            ))
            .expect("open dataset");

        assert_eq!(dataset.name(), "di-osc/calls");
        assert_eq!(dataset.version(), "v1");
        assert_eq!(dataset.license(), "Apache License 2.0");
        assert_eq!(dataset.snapshot_path(), Some(snapshot_path.as_path()));
        assert!(dataset.train().is_some());
        assert!(dataset.val().is_none());
        assert!(dataset.test().is_some());
        assert!(dataset.train_database_path().is_some());
        assert!(dataset.val_database_path().is_none());
        assert!(dataset.test_database_path().is_some());
        assert_eq!(
            *downloader.request.lock().expect("request lock"),
            Some(DownloadRequest {
                repo_id: "di-osc/calls".to_owned(),
                revision: "v1".to_owned(),
                cache_dir,
            })
        );
        assert!(
            dataset
                .train()
                .expect("train database")
                .set_metadata("changed", &true.into())
                .is_err()
        );

        drop(dataset);
        fs::remove_dir_all(snapshot_path).expect("remove snapshot");
    }

    #[test]
    fn snapshot_without_readme_or_split_databases_is_valid() {
        let snapshot_path = temp_snapshot("empty-snapshot");
        fs::create_dir_all(&snapshot_path).expect("create snapshot");
        let downloader = FakeDownloader {
            snapshot_path: snapshot_path.clone(),
            request: Mutex::new(None),
        };
        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        let dataset = runtime
            .block_on(AudioDataset::from_modelscope_with_downloader(
                &downloader,
                "di-osc/empty",
                None,
                None,
            ))
            .expect("empty snapshot is valid");

        assert_eq!(dataset.license(), "");
        assert!(dataset.train().is_none());
        assert!(dataset.val().is_none());
        assert!(dataset.test().is_none());

        fs::remove_dir_all(snapshot_path).expect("remove snapshot");
    }

    #[test]
    fn modelscope_license_supports_plain_and_quoted_scalars() {
        assert_eq!(
            parse_modelscope_license("---\nlicense: Apache License 2.0\n---\n"),
            Ok("Apache License 2.0".to_owned())
        );
        assert_eq!(
            parse_modelscope_license("---\nlicense: \"CC-BY-4.0\"\n---\n"),
            Ok("CC-BY-4.0".to_owned())
        );
        assert_eq!(
            parse_modelscope_license("---\nlicense: 'ODC-BY'\n---\n"),
            Ok("ODC-BY".to_owned())
        );
        assert_eq!(
            parse_modelscope_license("# No front matter\n"),
            Ok(String::new())
        );
        assert_eq!(
            parse_modelscope_license("---\nlanguage: zh\n---\n"),
            Ok(String::new())
        );
        assert!(parse_modelscope_license("---\nlicense: [MIT]\n---\n").is_err());
        assert!(parse_modelscope_license("---\nlicense: MIT\n").is_err());
    }

    #[test]
    fn invalid_present_split_database_has_a_contextual_error() {
        let snapshot_path = temp_snapshot("invalid-snapshot");
        fs::create_dir_all(&snapshot_path).expect("create snapshot");
        fs::write(snapshot_path.join(VAL_DB_FILE), "not sqlite").expect("write invalid database");
        let downloader = FakeDownloader {
            snapshot_path: snapshot_path.clone(),
            request: Mutex::new(None),
        };
        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        let error = runtime
            .block_on(AudioDataset::from_modelscope_with_downloader(
                &downloader,
                "di-osc/broken",
                None,
                None,
            ))
            .expect_err("invalid database should fail");

        assert!(matches!(
            &error,
            AudioDatasetError::Database { split, .. } if *split == "val"
        ));
        assert!(error.to_string().contains("val"));

        fs::remove_dir_all(snapshot_path).expect("remove snapshot");
    }

    #[test]
    fn modelscope_download_failure_has_repository_context() {
        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        let error = runtime
            .block_on(AudioDataset::from_modelscope_with_downloader(
                &FailingDownloader,
                "di-osc/missing",
                Some("v2"),
                None,
            ))
            .expect_err("download should fail");

        assert!(matches!(
            &error,
            AudioDatasetError::ModelScopeDownload {
                repo_id,
                revision,
                ..
            } if repo_id == "di-osc/missing" && revision == "v2"
        ));
    }
}
