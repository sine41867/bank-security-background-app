-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Nov 17, 2024 at 11:28 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_security_system`
--

-- --------------------------------------------------------

--
-- Table structure for table `tbl_alerts`
--

CREATE TABLE `tbl_alerts` (
  `alert_id` int(11) NOT NULL,
  `type` varchar(15) NOT NULL,
  `description` varchar(200) NOT NULL,
  `photo` longblob DEFAULT NULL,
  `time` datetime NOT NULL,
  `branch_id` varchar(8) NOT NULL,
  `generated_by` varchar(8) DEFAULT NULL,
  `checked_by` varchar(8) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_blacklisted`
--

CREATE TABLE `tbl_blacklisted` (
  `cif_no` varchar(8) NOT NULL,
  `photo` longblob NOT NULL,
  `description` varchar(200) NOT NULL,
  `inserted_by` varchar(8) NOT NULL,
  `time` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_robbers`
--

CREATE TABLE `tbl_robbers` (
  `robber_id` varchar(8) NOT NULL,
  `name` varchar(100) NOT NULL,
  `photo` longblob NOT NULL,
  `description` varchar(200) NOT NULL,
  `inserted_by` varchar(8) NOT NULL,
  `time` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_users`
--

CREATE TABLE `tbl_users` (
  `user_id` varchar(8) NOT NULL,
  `password` varchar(255) NOT NULL,
  `type` int(11) NOT NULL,
  `inserted_by` varchar(8) NOT NULL,
  `time` datetime NOT NULL,
  `invalid_attempts` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `tbl_alerts`
--
ALTER TABLE `tbl_alerts`
  ADD PRIMARY KEY (`alert_id`),
  ADD KEY `tbl_alerts_ibfk_1` (`generated_by`);

--
-- Indexes for table `tbl_blacklisted`
--
ALTER TABLE `tbl_blacklisted`
  ADD PRIMARY KEY (`cif_no`),
  ADD KEY `inserted_by` (`inserted_by`);

--
-- Indexes for table `tbl_robbers`
--
ALTER TABLE `tbl_robbers`
  ADD PRIMARY KEY (`robber_id`),
  ADD KEY `inserted_by` (`inserted_by`);

--
-- Indexes for table `tbl_users`
--
ALTER TABLE `tbl_users`
  ADD PRIMARY KEY (`user_id`),
  ADD KEY `inserted_by` (`inserted_by`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `tbl_alerts`
--
ALTER TABLE `tbl_alerts`
  MODIFY `alert_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `tbl_blacklisted`
--
ALTER TABLE `tbl_blacklisted`
  ADD CONSTRAINT `tbl_blacklisted_ibfk_1` FOREIGN KEY (`inserted_by`) REFERENCES `tbl_users` (`user_id`) ON DELETE NO ACTION ON UPDATE NO ACTION;

--
-- Constraints for table `tbl_robbers`
--
ALTER TABLE `tbl_robbers`
  ADD CONSTRAINT `tbl_robbers_ibfk_1` FOREIGN KEY (`inserted_by`) REFERENCES `tbl_users` (`user_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
