-- ============================================================
-- HSC-SSP PDR3  |  Step 2: Epoch list + coadd photometry per source
-- Schema: pdr3_dud  (Deep+UltraDeep, covers COSMOS)
--
-- For each source we retrieve:
--   • Coadd forced magnitudes in g, r, i, z, y (from the forced table)
--   • The list of individual visits that contributed to each filter coadd
--     (via mosaicframe + frame), with MJD, seeing, zero-point, etc.
--
-- NOTE: The magnitudes here are COADD magnitudes — the same value
-- is repeated for every visit row of the same source in the same band.
-- They represent the stacked measurement across all contributing visits.
--
-- Paste in CAS at: https://hsc-release.mtk.nao.ac.jp/datasearch/
-- Use "Enqueue" — result will be hundreds to thousands of rows.
-- ============================================================

WITH source_ids AS (

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_42'   AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.179522, 150.180078, 2.110062, 2.110618)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_268'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.087163, 150.087719, 1.741656, 1.742212)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_346'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.930592, 149.931148, 2.118559, 2.119115)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_349'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.004102, 150.004658, 2.038700, 2.039256)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_451'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.002292, 150.002848, 2.258385, 2.258941)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_563'  AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.920262, 149.920818, 2.543390, 2.543946)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_1205' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 150.010412, 150.010968, 2.332723, 2.333279)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_1605' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.832314, 149.832870, 2.710581, 2.711137)
    AND isprimary

  UNION ALL

  SELECT f.object_id, f.tract, f.patch, f.ra, f.dec,
         f.g_cmodel_mag, f.g_cmodel_magerr,
         f.r_cmodel_mag, f.r_cmodel_magerr,
         f.i_cmodel_mag, f.i_cmodel_magerr,
         f.z_cmodel_mag, f.z_cmodel_magerr,
         f.y_cmodel_mag, f.y_cmodel_magerr,
         'cid_2550' AS name
  FROM pdr3_dud.forced AS f
  WHERE boxSearch(coord, 149.874252, 149.874808, 2.361211, 2.361767)
    AND isprimary

)

SELECT DISTINCT ON (s.name, fr.visit, fr.filter)
  s.name,
  s.object_id,
  s.ra,
  s.dec,
  s.tract,
  s.patch,
  -- Coadd magnitudes (stacked across all visits in that filter)
  s.g_cmodel_mag,  s.g_cmodel_magerr,
  s.r_cmodel_mag,  s.r_cmodel_magerr,
  s.i_cmodel_mag,  s.i_cmodel_magerr,
  s.z_cmodel_mag,  s.z_cmodel_magerr,
  s.y_cmodel_mag,  s.y_cmodel_magerr,
  -- Per-visit metadata
  fr.visit,
  fr.ccd,
  fr.filter,
  fr.pointing,
  fr.mjd,
  fr.exptime,
  fr.seeing,
  fr.zeropt,
  fr.maglimit
FROM source_ids s
JOIN pdr3_dud.mosaicframe mf
  ON mf.tract = s.tract
 AND mf.patch = s.patch
JOIN pdr3_dud.frame fr
  ON fr.visit = mf.visit
 AND fr.ccd  = mf.ccd
WHERE fr.filter IN ('HSC-G', 'HSC-R', 'HSC-R2', 'HSC-I', 'HSC-I2', 'HSC-Z', 'HSC-Y')
ORDER BY s.name, fr.filter, fr.visit, fr.ccd
;
