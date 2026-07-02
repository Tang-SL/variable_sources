-- ============================================================
-- HSC-SSP PDR3  |  Step 2: Per-epoch light curves for all CID sources
-- Schema: pdr3_dud  (Deep+UltraDeep, covers COSMOS)
--
-- boxSearch(coord, ra1, ra2, dec1, dec2) — all in DEGREES
-- Each box is ±1 arcsec (0.000278 deg) around each CID position.
--
-- Paste in CAS at: https://hsc-release.mtk.nao.ac.jp/datasearch/
-- Use "Enqueue" (not Preview) — result will be thousands of rows.
-- ============================================================

WITH source_ids AS (

  SELECT f.object_id, 'cid_42'   AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.179522, 150.180078, 2.110062, 2.110618)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_268'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.087163, 150.087719, 1.741656, 1.742212)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_346'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.930592, 149.931148, 2.118559, 2.119115)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_349'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.004102, 150.004658, 2.038700, 2.039256)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_451'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.002292, 150.002848, 2.258385, 2.258941)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_563'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.920262, 149.920818, 2.543390, 2.543946)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_1205' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.010412, 150.010968, 2.332723, 2.333279)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_1605' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.832314, 149.832870, 2.710581, 2.711137)
    AND isprimary

  UNION ALL

  SELECT f.object_id, 'cid_2550' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.874252, 149.874808, 2.361211, 2.361767)
    AND isprimary

)

SELECT
  s.name,
  s.object_id,
  fi.filter,
  fi.mjd,
  fi.exptime,
  fi.seeing,
  f2.id        AS frame_id,
  f2.psf_flux,
  f2.psf_flux_err,
  flux_to_mag(f2.psf_flux)                     AS psf_mag,
  flux_to_magerr(f2.psf_flux, f2.psf_flux_err) AS psf_magerr,
  f2.pixelflags_bad,
  f2.pixelflags_saturatedcenter,
  f2.pixelflags_cr,
  f2.pixelflags_edge,
  f2.pixelflags_interpolatedcenter
FROM source_ids s
JOIN pdr3_dud.forced2 f2 ON s.object_id = f2.object_id
JOIN pdr3_dud.frame   fi ON f2.id = fi.id
WHERE f2.psf_flux_err > 0
ORDER BY s.name, fi.filter, fi.mjd;
