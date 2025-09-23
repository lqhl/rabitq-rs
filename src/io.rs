use std::convert::TryInto;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::path::Path;

fn read_vecs_from_reader<R, T, F>(
    mut reader: R,
    limit: Option<usize>,
    convert: F,
) -> io::Result<Vec<Vec<T>>>
where
    R: Read,
    F: Fn([u8; 4]) -> T,
{
    let mut vectors = Vec::new();

    loop {
        if let Some(max) = limit {
            if vectors.len() >= max {
                break;
            }
        }

        let mut dim_buf = [0u8; 4];
        match reader.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }
        let dim = i32::from_le_bytes(dim_buf);
        if dim < 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("negative dimension {dim} encountered"),
            ));
        }
        let dim = dim as usize;

        let mut buffer = vec![0u8; dim * 4];
        reader.read_exact(&mut buffer)?;
        if !buffer.chunks_exact(4).remainder().is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "vector payload is not aligned to 4-byte elements",
            ));
        }

        let mut vector = Vec::with_capacity(dim);
        for chunk in buffer.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().expect("chunk of length 4");
            vector.push(convert(bytes));
        }
        vectors.push(vector);
    }

    Ok(vectors)
}

pub fn read_fvecs_from_reader<R: Read>(
    reader: R,
    limit: Option<usize>,
) -> io::Result<Vec<Vec<f32>>> {
    read_vecs_from_reader(reader, limit, f32::from_le_bytes)
}

pub fn read_ivecs_from_reader<R: Read>(
    reader: R,
    limit: Option<usize>,
) -> io::Result<Vec<Vec<i32>>> {
    read_vecs_from_reader(reader, limit, i32::from_le_bytes)
}

pub fn read_fvecs<P: AsRef<Path>>(path: P, limit: Option<usize>) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    read_fvecs_from_reader(BufReader::new(file), limit)
}

pub fn read_ivecs<P: AsRef<Path>>(path: P, limit: Option<usize>) -> io::Result<Vec<Vec<i32>>> {
    let file = File::open(path)?;
    read_ivecs_from_reader(BufReader::new(file), limit)
}

pub fn read_ids_from_reader<R: Read>(reader: R, limit: Option<usize>) -> io::Result<Vec<usize>> {
    let rows = read_ivecs_from_reader(reader, limit)?;
    convert_ids(rows)
}

pub fn read_ids<P: AsRef<Path>>(path: P, limit: Option<usize>) -> io::Result<Vec<usize>> {
    let file = File::open(path)?;
    read_ids_from_reader(BufReader::new(file), limit)
}

pub fn read_groundtruth_from_reader<R: Read>(
    reader: R,
    limit: Option<usize>,
) -> io::Result<Vec<Vec<usize>>> {
    let rows = read_ivecs_from_reader(reader, limit)?;
    convert_rows(rows)
}

pub fn read_groundtruth<P: AsRef<Path>>(
    path: P,
    limit: Option<usize>,
) -> io::Result<Vec<Vec<usize>>> {
    let file = File::open(path)?;
    read_groundtruth_from_reader(BufReader::new(file), limit)
}

fn convert_ids(rows: Vec<Vec<i32>>) -> io::Result<Vec<usize>> {
    let mut ids = Vec::with_capacity(rows.len());
    for (idx, mut row) in rows.into_iter().enumerate() {
        if row.is_empty() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("cluster id row {idx} is empty"),
            ));
        }
        if row.len() > 1 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "cluster id row {idx} has {} elements, expected exactly 1",
                    row.len()
                ),
            ));
        }
        let value = row.remove(0);
        if value < 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("cluster id {value} at row {idx} is negative"),
            ));
        }
        ids.push(value as usize);
    }
    Ok(ids)
}

fn convert_rows(rows: Vec<Vec<i32>>) -> io::Result<Vec<Vec<usize>>> {
    let mut converted = Vec::with_capacity(rows.len());
    for (row_idx, row) in rows.into_iter().enumerate() {
        let mut converted_row = Vec::with_capacity(row.len());
        for value in row {
            if value < 0 {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    format!("ground truth entry {value} in row {row_idx} is negative"),
                ));
            }
            converted_row.push(value as usize);
        }
        converted.push(converted_row);
    }
    Ok(converted)
}
