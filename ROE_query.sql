WITH AT_RawData AS
(
    SELECT
        A.[NRI_CODE],
        A.[FTERM],
        A.[CTYPE],
        A.[SETTLEMENT_FLG],
        A.[FDATE],
        A.[NMONTH],
        -- FDATE（YYYYMMDD）をdate型に変換
        TRY_CONVERT
        (
            date,
            RIGHT
            (
                '00000000' + CONVERT(varchar(8), A.[FDATE]),
                8
            ),
            112 -- 日付形式であることの明示
        ) AS 決算年月日

    FROM [IDSQE].[dbo].[NRIFIN_CON_AT4] AS A
),

AT_Data AS
(
    SELECT
        A.[NRI_CODE],
        A.[FTERM],
        A.[CTYPE],
        A.[SETTLEMENT_FLG],
        A.[FDATE],
        A.[NMONTH],
        A.決算年月日,

        /*
          会計年度開始日を計算
          FDATEからNMONTHか月前の月末を求め、その翌日を会計年度開始日とする

          例：2024年3月31日・NMONTH=12
              EOMONTH(2024-03-31, -12) = 2023-03-31 -> その翌日 = 2023-04-01
        */
        DATEADD
        (
            DAY,
            1, -- 1日加える
            EOMONTH(A.決算年月日, -A.[NMONTH]) -- 月末からNMONTHか月戻る
        ) AS 会計年度開始日,

        -- フラグに従い、本中区分を四半期番号に変換
        CASE A.[SETTLEMENT_FLG]
            WHEN 3 THEN 1 -- 第1四半期決算
            WHEN 2 THEN 2 -- 中間・第2四半期決算
            WHEN 4 THEN 3 -- 第3四半期決算
            WHEN 1 THEN 4 -- 本決算・第4四半期
        END AS 四半期番号

    FROM AT_RawData AS A

    WHERE A.決算年月日 IS NOT NULL

    -- 通常の3か月単位の四半期決算だけを対象とする -> 決算期の変更を除く
    AND
    (
        (A.[SETTLEMENT_FLG] = 3 AND A.[NMONTH] = 3)
        OR (A.[SETTLEMENT_FLG] = 2 AND A.[NMONTH] = 6)
        OR (A.[SETTLEMENT_FLG] = 4 AND A.[NMONTH] = 9)
        OR (A.[SETTLEMENT_FLG] = 1 AND A.[NMONTH] = 12)
    )
),

-- 企業、決算期、会計基準、累積四半期純利益を取得
PL_BaseData AS
(
    SELECT
        P.[NRI_CODE],
        P.[FTERM] AS 決算期,
        LEFT(CONVERT(varchar(6), P.[FTERM]), 4) AS 決算年,

        A.[SETTLEMENT_FLG] AS 本中区分,
        A.四半期番号,
        P.[CTYPE] AS 会計基準,
        P.[NETINCM] AS 累積当期純利益,

        A.[FDATE],
        A.決算年月日,
        A.[NMONTH] AS 決算月数,
        A.会計年度開始日

    FROM [IDSQE].[dbo].[NRIFIN_CON_PL4] AS P

    JOIN AT_Data AS A
        ON  P.[NRI_CODE] = A.[NRI_CODE]
        AND P.[FTERM]    = A.[FTERM]
        AND P.[CTYPE]    = A.[CTYPE] -- 会計基準が一致しているもの

    -- 純利益の欠損値を除外
    WHERE P.[NETINCM] IS NOT NULL
      AND P.[NETINCM] NOT IN (-999999998, -999999999)
),

-- 各四半期の累積利益を記載
PL_PeriodData AS
(
    SELECT
        *,

        -- 第1四半期累積利益
        MAX
        (
            CASE
                WHEN 本中区分 = 3
                    THEN 累積当期純利益
            END
        ) OVER
        (
            -- 決算年ではなく、会計年度開始日でグループ化
            PARTITION BY
                NRI_CODE,
                会計年度開始日,
                会計基準
        ) AS 第1四半期累積利益,

        -- 第2四半期（半期）累積利益
        MAX
        (
            CASE
                WHEN 本中区分 = 2
                    THEN 累積当期純利益
            END
        ) OVER
        (
            PARTITION BY
                NRI_CODE,
                会計年度開始日,
                会計基準
        ) AS 中間累積利益,

        -- 第3四半期累積利益
        MAX
        (
            CASE
                WHEN 本中区分 = 4
                    THEN 累積当期純利益
            END
        ) OVER
        (
            PARTITION BY
                NRI_CODE,
                会計年度開始日,
                会計基準
        ) AS 第3四半期累積利益
    FROM PL_BaseData
    -- 第4はそのまま収録値を用いれば良い
),

-- 四半期単独利益の計算
PL_Data AS
(
    SELECT
        NRI_CODE,
        決算期,
        決算年,
        決算年月日,
        決算月数,
        会計年度開始日,
        本中区分,
        四半期番号,
        会計基準,
        累積当期純利益,

        -- 差分を用いて単独利益を計算
        CASE 本中区分
            -- Q1単独利益 = Q1累積利益
            -- 差分不要
            WHEN 3
                THEN 累積当期純利益

            --Q2単独利益 = Q2累積利益 - Q1累積利益
            --Q1が欠損している場合はNULLとする
            WHEN 2
                THEN CASE
                    WHEN 第1四半期累積利益 IS NOT NULL
                        THEN 累積当期純利益 - 第1四半期累積利益
                    ELSE NULL
                END

            /* Q3単独利益 = Q3累積利益 - Q2累積利益
               Q2が欠損している場合はNULL */
            WHEN 4
                THEN CASE
                    WHEN 中間累積利益 IS NOT NULL
                        THEN 累積当期純利益 - 中間累積利益
                    ELSE NULL
                END

            /* Q4単独利益 = 本決算累積利益 - Q3累積利益
               Q3が欠損している場合はNULLとする。*/
            WHEN 1
                THEN CASE
                    WHEN 第3四半期累積利益 IS NOT NULL
                        THEN 累積当期純利益 - 第3四半期累積利益
                    ELSE NULL
                END

            ELSE NULL
        END AS 四半期単独当期純利益,

        /*
          四半期単独利益に対応する前四半期末日を計算
          会計年度開始日に「NMONTH - 3」か月を加え、その前日を前四半期末日とする

          例：2024年3月期
              Q1 (NMONTH=3)
                  2023-04-01 + 0か月 - 1日
                  = 2023-03-31

              Q2 (NMONTH=6)
                  2023-04-01 + 3か月 - 1日
                  = 2023-06-30

              Q3 (NMONTH=9)
                  2023-04-01 + 6か月 - 1日
                  = 2023-09-30

              Q4 (NMONTH=12)
                  2023-04-01 + 9か月 - 1日
                  = 2023-12-31
        */
        DATEADD
        (
            DAY,
            -1,
            DATEADD
            (
                MONTH,
                決算月数 - 3,
                会計年度開始日
            )
        ) AS 前四半期末日

    FROM PL_PeriodData
),

-- B/Sデータを抽出
BS_Data AS
(
    SELECT
        B.[NRI_CODE],
        B.[FTERM] AS 決算期,
        LEFT(CONVERT(varchar(6), B.[FTERM]), 4) AS 決算年,
        A.[SETTLEMENT_FLG] AS 本中区分,
        B.[CTYPE] AS 会計基準,

        A.[FDATE],
        A.決算年月日,
        A.[NMONTH] AS 決算月数,

        B.[NETEQTY] AS 自己資本,

        CASE
            -- DTLの異常値またはNULLを0とする
            WHEN B.[DETAXLI] IS NULL
              OR B.[DETAXLI] IN (-999999998, -999999999)
                THEN 0
            ELSE B.[DETAXLI]
        END AS 繰延税金負債,

        -- BEQの定義（自己資本+DTL）
        B.[NETEQTY]
        + CASE
            -- DTLの欠損値を0とする
            WHEN B.[DETAXLI] IS NULL
              OR B.[DETAXLI] IN (-999999998, -999999999)
                THEN 0
            ELSE B.[DETAXLI]
        END AS BEQ

    FROM [IDSQE].[dbo].[NRIFIN_CON_BS4] AS B

    /*
      B/Sは期末時点の残高であり、決算期間の長さ（NMONTH）とは関係がない。
      そのため、NMONTHで絞り込んだAT_Dataではなく、絞り込み前のAT_RawDataと結合する。
      （AT_Dataと結合すると、前期が決算期変更の年の場合に、翌年Q1の分母が見つからなくなる）
    */
    JOIN AT_RawData AS A
        ON  B.[NRI_CODE] = A.[NRI_CODE]
        AND B.[FTERM]    = A.[FTERM]
        AND B.[CTYPE]    = A.[CTYPE]

    -- 自己資本が欠損の企業は除外
    WHERE A.決算年月日 IS NOT NULL
      AND B.[NETEQTY] IS NOT NULL AND B.[NETEQTY] NOT IN (-999999998, -999999999)
),

-- ROEの計算
ROE_Data AS
(
    SELECT
        PL.NRI_CODE,
        PL.決算期 AS 当期決算期,
        PL.決算年 AS 当期決算年,
        PL.決算年月日 AS 当期決算年月日,
        PL.決算月数 AS 当期決算月数,
        PL.会計年度開始日,
        PL.本中区分,
        PL.四半期番号,
        PL.会計基準 AS 当期会計基準,
        PL.累積当期純利益,
        PL.四半期単独当期純利益 AS profit_t,
        PL.前四半期末日,

        -- 前四半期のB/Sデータ項目
        BS.決算期 AS 前期決算期,
        BS.決算年 AS 前期決算年,
        BS.決算年月日 AS 前期決算年月日,
        BS.本中区分 AS 前期本中区分,
        BS.会計基準 AS 前期会計基準,
        BS.自己資本 AS equity_t_minus_1,
        BS.繰延税金負債 AS DTL_t_minus_1,
        BS.BEQ AS BEQ_t_minus_1,

        -- ROEの計算式:=純利益(t)/BEQ(t-1)
        CAST(PL.四半期単独当期純利益 AS decimal(38, 6))
        /
        NULLIF(CAST(BS.BEQ AS decimal(38, 6)),0) AS ROE -- 1期前のB/Sの値

    FROM PL_Data AS PL

    --LEFT JOINとすることで、前四半期B/Sが見つからない行もROE_Data上では確認できるようにする。
    --最終出力では、BEQが取得できた行だけを抽出
    LEFT JOIN BS_Data AS BS
        ON  BS.NRI_CODE = PL.NRI_CODE
        -- 会計基準の合致を条件とする
        AND BS.会計基準 = PL.会計基準

        /*
          B/Sの決算年月日が、P/Lに対応する前四半期末日と一致するものを結合
          FTERMを単純に3か月前へ移動するのではなく、FDATEとNMONTHから算出した前四半期末日を使用
        */
        AND BS.決算年月日 = PL.前四半期末日
),

/*
  会計基準が複数ある場合の絞り込み
  経過措置などで、同じ企業・同じ四半期に複数の会計基準の数値がある場合は、1行に絞る。

  1. ROEが計算できた行（分子・分母が同じ基準でそろっている行）だけを残す
  2. 複数残った場合は、IFRS -> 日本基準 -> SEC基準 の順で優先する

  四半期によって採用する基準が変わることはありうるが、
  1つのROEの中では、分子と分母の会計基準は必ずそろっている。
*/
ROE_Ranked AS
(
    SELECT
        R.*,

        ROW_NUMBER() OVER
        (
            PARTITION BY
                R.NRI_CODE,
                R.当期決算年月日 -- 1企業・1四半期
            ORDER BY
                CASE R.当期会計基準
                    WHEN '3' THEN 1 -- IFRS
                    WHEN '2' THEN 2 -- 日本基準
                    WHEN '1' THEN 3 -- SEC基準
                    ELSE 4
                END
        ) AS 基準優先順位

    FROM ROE_Data AS R

    WHERE R.profit_t IS NOT NULL
      AND R.BEQ_t_minus_1 IS NOT NULL
      -- 自己資本がゼロ以下（債務超過）の企業は除外する（ROEの符号が逆転するため）
      AND R.BEQ_t_minus_1 > 0
)

SELECT
    NRI_CODE,
    当期決算期,
    当期決算年,
    当期決算年月日,
    当期決算月数,
    会計年度開始日,
    本中区分,
    四半期番号,

    CASE 本中区分
        WHEN 1 THEN N'本決算・第4四半期'
        WHEN 2 THEN N'中間決算・第2四半期'
        WHEN 3 THEN N'第1四半期決算'
        WHEN 4 THEN N'第3四半期決算'
        ELSE N'不明'
    END AS 決算区分,

    CASE 当期会計基準
        WHEN '1' THEN N'SEC式'
        WHEN '2' THEN N'日本式'
        WHEN '3' THEN N'IFRS'
        ELSE N'不明'
    END AS 会計基準名称,

    当期会計基準,
    累積当期純利益,
    profit_t,

    前四半期末日,
    前期決算期,
    前期決算年月日,
    前期会計基準,

    equity_t_minus_1,
    DTL_t_minus_1,
    BEQ_t_minus_1,

    --四半期ROE, 3か月間の利益を分子としているため、年率でない
    ROE
    /*
    ,

    -- ROE * 100 AS ROEパーセント,


    -- 必要に応じて使用する単純年率換算値。
    -- 通常の3か月決算のみを対象としているため4倍する。

    ROE * 4 AS 年率換算ROE,
    ROE * 4 * 100 AS 年率換算ROEパーセント
    */
FROM ROE_Ranked

-- 欠損値・ゼロ以下の自己資本の除外は ROE_Ranked で実施済み
-- 1企業・1四半期につき、優先順位が最も高い会計基準の行だけを抽出
WHERE 基準優先順位 = 1

ORDER BY
    NRI_CODE,
    会計年度開始日,
    当期会計基準,
    四半期番号;
