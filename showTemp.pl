#!/usr/bin/perl -w
# By "peba@inode.at"

use Tk;
$main=MainWindow->new;
$main->resizable (0,0);
$temp="--";

$myframe=$main->Frame ();
$myframe->Label (-text => 'Temperature:')->pack ();
$temp_label=$myframe->Label (-text => "$temp")->pack ();
$myframe->pack();

$main->repeat(1000,\&update_temp_label);
MainLoop;

sub update_temp_label {
    $temp=`cat /sys/devices/virtual/thermal/thermal_zone0/temp`;
    $temp_label->configure('-text' => $temp);
}